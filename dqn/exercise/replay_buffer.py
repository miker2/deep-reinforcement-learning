import torch
import numpy as np
import random
from collections import namedtuple, deque

class SumTree:
    """
    A binary tree data structure where the value of a parent node is the sum
    of its children. This allows for efficient O(log n) sampling based on priorities.
    """
    write_idx = 0 # Tracks the next position to write data in the tree leaves

    def __init__(self, capacity):
        self.capacity = capacity
        self.sum_tree = np.zeros(2 * capacity - 1)
        self.min_tree = np.zeros(2 * capacity - 1)
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.sum_tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.sum_tree):
            return idx
        if s <= self.sum_tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.sum_tree[left])

    def total(self):
        """Return the total priority of the tree."""
        return self.sum_tree[0]

    def add(self, p):
        idx = self.write_idx + self.capacity - 1
        self.update(idx, p)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, p):
        if idx < self.capacity -1:
             raise ValueError("Index must point to a leaf node.")
        if idx >= len(self.sum_tree):
             raise IndexError("Index out of tree bounds.")
        change = p - self.sum_tree[idx]
        self.sum_tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.sum_tree[idx], dataIdx)


class PrioritizedReplayBuffer:
    """
    Fixed-size buffer to store experience tuples (with 1-D vector states),
    sampled based on priority. Uses a SumTree for efficient priority updates
    and sampling.
    """
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed,
                 device, discrete_states=False, discrete_actions=True):
        """
        Initialize a PrioritizedReplayBuffer object for 1-D vector states.

        Args:
            state_size (int): dimension of the 1-D state vector
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (str): device to store tensors on ('cpu' or 'cuda')
        """
        self.action_size = action_size
        self.state_size = state_size # Store state dimension
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device
        self.ptr = 0
        self.size = 0

        state_type = np.int64 if discrete_states else np.float32
        action_type = np.int64 if discrete_actions else np.float32

        # Storage arrays for experiences
        self.states = np.zeros((buffer_size, state_size), dtype=action_type)
        self.actions = np.zeros((buffer_size, action_size), dtype=state_type)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_size), dtype=state_type)

        self.dones = np.zeros((buffer_size, 1), dtype=np.int8)

        self.tree = SumTree(buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        """Adds a new experience to memory and updates its priority."""
        max_priority = np.max(self.tree.sum_tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0

        # Store experience - ensure types match numpy arrays
        self.states[self.ptr] = np.float32(state)
        self.actions[self.ptr] = np.int64(action)
        self.rewards[self.ptr] = np.float32(reward)
        self.next_states[self.ptr] = np.float32(next_state)
        self.dones[self.ptr] = np.float32(done)

        self.tree.add(max_priority)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)


    def sample(self):
        """Samples a batch of experiences based on priority."""
        if self.size < self.batch_size:
             print(f"Warning: Buffer size ({self.size}) is less than batch size ({self.batch_size}). Sampling all available elements.")
             current_batch_size = self.size
        else:
             current_batch_size = self.batch_size

        # --- MODIFIED SECTION ---
        # Adjust batch array shapes and types
        batch_states = np.zeros((current_batch_size, self.state_size), dtype=np.float32)
        batch_actions = np.zeros((current_batch_size, 1), dtype=np.int64) # Match self.actions
        batch_rewards = np.zeros((current_batch_size, 1), dtype=np.float32) # Match self.rewards
        batch_next_states = np.zeros((current_batch_size, self.state_size), dtype=np.float32)
        batch_dones = np.zeros((current_batch_size, 1), dtype=np.float32) # Match self.dones
        # --- END MODIFIED SECTION ---

        idxs = np.empty((current_batch_size,), dtype=np.int32)
        is_weights = np.empty((current_batch_size, 1), dtype=np.float32)

        priority_segment = self.tree.total() / current_batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.sum_tree[-self.tree.capacity:]) / self.tree.total()
        if min_prob == 0:
             min_prob = self.epsilon**self.alpha / self.tree.total() if self.tree.total() > 0 else 1.0

        if self.size > 0 and min_prob > 0 and self.tree.total() > 0:
             max_weight = (self.size * min_prob) ** (-self.beta)
        else:
             max_weight = 1.0


        for i in range(current_batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            # Handle edge case where total priority might be zero initially
            if self.tree.total() > 0:
                 s = random.uniform(a, b)
                 (tree_idx, priority, data_idx) = self.tree.get(s)
                 sampling_probability = priority / self.tree.total()
                 is_weights[i, 0] = (self.size * sampling_probability) ** (-self.beta)
            else: # If total is zero, sample uniformly (or handle as error)
                 data_idx = random.randint(0, self.size - 1) # Fallback: uniform random index
                 is_weights[i, 0] = 1.0 # Cannot calculate priority-based weight


            idxs[i] = data_idx
            batch_states[i] = self.states[data_idx]
            batch_actions[i] = self.actions[data_idx]
            batch_rewards[i] = self.rewards[data_idx]
            batch_next_states[i] = self.next_states[data_idx]
            batch_dones[i] = self.dones[data_idx]

        # Normalize IS weights only if max_weight is positive
        if max_weight > 0:
            is_weights /= max_weight
        else:
             print("Warning: max_weight is zero, cannot normalize IS weights.")


        # --- MODIFIED SECTION ---
        # Convert batch to PyTorch tensors - remove image-specific permute and scaling
        states_tensor = torch.from_numpy(batch_states).float().to(self.device)
        actions_tensor = torch.from_numpy(batch_actions).long().to(self.device) # Ensure actions are Long type for embedding layers or indexing
        rewards_tensor = torch.from_numpy(batch_rewards).float().to(self.device)
        next_states_tensor = torch.from_numpy(batch_next_states).float().to(self.device)
        dones_tensor = torch.from_numpy(batch_dones).float().to(self.device)
        is_weights_tensor = torch.from_numpy(is_weights).float().to(self.device)
        # --- END MODIFIED SECTION ---

        return (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor), idxs, is_weights_tensor

    def update_priorities(self, batch_indices, errors):
        """Updates priorities of sampled experiences."""
        if isinstance(errors, torch.Tensor):
            errors = errors.detach().cpu().numpy()

        errors = np.squeeze(errors)
        if errors.ndim == 0:
             errors = np.array([errors])

        if len(batch_indices) != len(errors):
             raise ValueError("Number of indices must match number of errors")

        for idx, error in zip(batch_indices, errors):
            # Ensure index is within the valid range of stored items
            if idx < 0 or idx >= self.buffer_size:
                 print(f"Warning: Attempting to update priority for invalid buffer index {idx}. Skipping.")
                 continue
            # Cannot update priority for an index that hasn't been filled yet
            # Note: This check is implicitly handled because `sample` only returns valid indices up to `self.size`
            # if idx >= self.size: # This check might be overly cautious depending on SumTree logic
            #      print(f"Warning: Attempting to update priority for index {idx} beyond current size {self.size}. Skipping.")
            #      continue

            priority = self._get_priority(error)
            tree_idx = idx + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size

# --- Example Usage (Modified for 1-D Vector States) ---
if __name__ == '__main__':
    # Example Configuration
    STATE_DIM = 10 # Dimension of the 1-D state vector
    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    ACTION_SIZE = 4
    SEED = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Create the buffer (pass state_dim)
    buffer = PrioritizedReplayBuffer(ACTION_SIZE, STATE_DIM, BUFFER_SIZE, BATCH_SIZE, SEED, DEVICE)

    # --- Dummy Data Generation (1-D vectors) ---
    def create_dummy_experience(state_dim, action_size):
        state = np.random.randn(state_dim).astype(np.float32) # Random float vector
        action = random.randint(0, action_size - 1)
        reward = random.uniform(-1, 1)
        next_state = np.random.randn(state_dim).astype(np.float32) # Random float vector
        done = random.choice([True, False])
        return state, action, reward, next_state, done

    # --- Fill buffer partially ---
    print("Adding experiences...")
    num_to_add = BUFFER_SIZE // 2
    for i in range(num_to_add):
        exp = create_dummy_experience(STATE_DIM, ACTION_SIZE)
        buffer.add(*exp)
        if (i+1) % 1000 == 0:
             print(f"Added {i+1}/{num_to_add} experiences...")


    print(f"Buffer size: {len(buffer)}")
    print(f"SumTree total priority: {buffer.tree.total():.2f}")

    # --- Sample a batch ---
    if len(buffer) >= BATCH_SIZE:
        print("\nSampling a batch...")
        experiences, idxs, is_weights = buffer.sample()
        states, actions, rewards, next_states, dones = experiences

        print(f"Sampled indices: {idxs[:5]}...")
        print(f"Sampled IS weights: {is_weights[:5].squeeze().cpu().numpy()}...")
        print(f"Sampled states shape: {states.shape}") # Should be [BATCH_SIZE, STATE_DIM]
        print(f"Sampled actions shape: {actions.shape}")
        print(f"Sampled rewards shape: {rewards.shape}")
        print(f"Sampled next_states shape: {next_states.shape}") # Should be [BATCH_SIZE, STATE_DIM]
        print(f"Sampled dones shape: {dones.shape}")
        print(f"Tensors on device: {states.device}")

        # --- Simulate TD errors and update priorities ---
        print("\nUpdating priorities...")
        dummy_td_errors = np.random.rand(BATCH_SIZE) * 10
        print(f"Dummy TD Errors (first 5): {dummy_td_errors[:5]}")

        buffer.update_priorities(idxs, dummy_td_errors)
        print("Priorities updated.")
        print(f"SumTree total priority after update: {buffer.tree.total():.2f}")

        # --- Sample again ---
        print("\nSampling another batch...")
        experiences2, idxs2, is_weights2 = buffer.sample()
        print(f"Sampled indices (second batch): {idxs2[:5]}...")
        print(f"Buffer beta value after sampling: {buffer.beta:.4f}")

    else:
        print(f"Buffer size {len(buffer)} is less than batch size {BATCH_SIZE}. Cannot sample a full batch yet.")