import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
# PER Hyperparameters
PER_ALPHA = 0.6         # prioritization exponent (0=uniform, 1=full)
PER_BETA = 0.4          # initial importance sampling exponent
PER_BETA_INCREMENT = 0.001 # beta annealing factor per sample step
PER_EPSILON = 1e-4      # small value added to priorities to ensure non-zero probability

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.backends.cuda.is_available():
    device = torch.cuda.current_device()
    print(torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print ("No accelerators found. Using CPU")
print(device)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,
                 hidden_layer_sizes=[256, 128, 64],
                 dropout_prob=0.25, epsilon=PER_EPSILON,
                 per_alpha=0., per_beta=0., per_beta_increment=0.):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            per_alpha (float): Alpha parameter for prioritized experience replay (0 = standard replay)
            per_beta (float): Beta parameter for prioritized experience replay (initial value)
            per_beta_increment (float): Beta increment per sample step
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.epsilon = epsilon


        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed,
                                       hidden_layer_sizes, dropout_prob).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed,
                                       hidden_layer_sizes, dropout_prob).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed,
                                                alpha=per_alpha, beta=per_beta,
                                                beta_increment_per_sampling=per_beta_increment,
                                                device=device) # Pass device

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def train(self):
        self.qnetwork_local.train()
        self.qnetwork_target.train()

    def eval(self):
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:

                # Sample with priorities, get experiences, weights, and indices
                experiences, weights, indices = self.memory.sample()
                self.learn(experiences, GAMMA, weights, indices)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, weights=None, indices=None):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            weights (torch.Tensor): importance sampling weights (for PER)
            indices (list): list of indices of sampled experiences (for PER)
        """
        states, actions, rewards, next_states, dones = experiences

        # Compute the Q-values for the given states and selected actions based on the
        # current network parameters
        q_values = self.qnetwork_local(states).gather(1, actions)

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            # Compute the Q-values for the next-states, given the optimal action (based
            # on the target network
            next_q_values = self.qnetwork_target(next_states).detach().max(1, keepdim=True)[0]
            # Compute Q targets for current states
            target_q_values = rewards + (gamma * next_q_values * (1 - dones))

        # Compute loss

        # Calculate element-wise loss
        elementwise_loss = F.mse_loss(q_values, target_q_values, reduction='none')
        # Apply importance sampling weights (weights are 1.0 if alpha=0)
        loss = (weights * elementwise_loss).mean()

        # Update priorities in the buffer (if using PER)
        if self.memory.alpha > 0:
            # Calculate absolute TD errors |Q_targets - Q_expected|
            td_errors = (target_q_values - q_values).abs().detach().cpu().numpy()
            # Add epsilon and update priorities
            new_priorities = td_errors + self.epsilon
            self.memory.update_priorities(indices, new_priorities.flatten())

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples and priorities."""

    def __init__(self, action_size, buffer_size, batch_size, seed,
                 alpha, beta, beta_increment_per_sampling, device):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): prioritization exponent (0=uniform, 1=full)
            beta (float): initial importance sampling exponent
            beta_increment_per_sampling (float): annealing factor for beta
            device (string): device to use for tensors ('cpu', 'cuda:0', 'mps')
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.device = device
        self.epsilon = PER_EPSILON # Use global constant

        # Use lists for memory and priorities, manage as circular buffer
        self.memory = [None] * buffer_size
        self.priorities = np.zeros(buffer_size)
        self.pos = 0 # Current insertion position
        self.current_size = 0 # Number of elements currently in buffer

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.max_priority = 0.0 # Initialize max priority

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with maximal priority."""
        e = self.experience(state, action, reward, next_state, done)
        # If alpha = 0, priorities don't matter for sampling, assign 1.0
        # If alpha > 0, assign max priority seen so far to ensure new experience
        priority = 1.0 if self.alpha == 0 else self.max_priority
        # Ensure priority is at least epsilon if alpha > 0
        priority = max(priority, self.epsilon) if self.alpha > 0 else priority

        self.memory[self.pos] = e
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.buffer_size
        if self.current_size < self.buffer_size:
            self.current_size += 1

    def sample(self):
        """Sample a batch of experiences from memory based on priorities."""
        if self.current_size == 0:
            return None, None, None # Or raise error

        if self.alpha == 0:
            # Uniform sampling
            indices = np.random.choice(self.current_size, self.batch_size, replace=True)
            weights = np.ones(self.batch_size) # All weights are 1.0
        else:
            # Prioritized sampling
            # Get priorities of existing experiences
            priorities_subset = self.priorities[:self.current_size]

            # Calculate probabilities P(i) = p_i^alpha / sum(p_k^alpha)
            scaled_priorities = priorities_subset ** self.alpha
            prob_sum = np.sum(scaled_priorities)
            if prob_sum == 0:  # Handle edge case and avoid division by zero
                prob_dist = np.ones(self.current_size) / self.current_size
            else:
                prob_dist = scaled_priorities / prob_sum

            # Sample indices based on probability distribution
            indices = np.random.choice(self.current_size, self.batch_size, p=prob_dist, replace=True)

            # Calculate Importance Sampling (IS) weights w_i = (N * P(i))^-beta / max(w_j)
            total_n = self.current_size
            weights = (total_n * prob_dist[indices]) ** (-self.beta)
            # Normalize weights by max weight for stability
            weights /= np.max(weights) if np.max(weights) > 0 else 1.0 # Avoid division by zero if max_weight is 0

            # Anneal beta towards 1.0
            self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # Retrieve experiences for sampled indices
        experiences = [self.memory[idx] for idx in indices]

        # Convert experiences to tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        # Convert weights to tensor
        weights_tensor = torch.from_numpy(weights).float().unsqueeze(1).to(self.device)

        return (states, actions, rewards, next_states, dones), weights_tensor, indices

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled experiences.

        Params
        ======
            indices (list): list of indices corresponding to experiences sampled
            priorities (np.array): array of new priorities (abs(TD error) + epsilon)
        """
        # Do nothing if alpha = 0
        if self.alpha == 0:
            return

        for idx, priority in zip(indices, priorities):
            # Ensure priority is positive and reflects the TD error + epsilon
            # The input 'priorities' should already be abs(TD error) + epsilon
            if idx < self.current_size: # Ensure index is valid
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority) # Update max priority seen

    def __len__(self):
        """Return the current size of internal memory."""
        return self.current_size