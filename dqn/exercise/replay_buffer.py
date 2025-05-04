import torch
import numpy as np
import random
from collections import namedtuple, deque

class SumTree:
    """
    A binary tree data structure where the value of a parent node is the sum
    of its children. This version also tracks the minimum value in the subtree
    rooted at each node. Allows for efficient O(log n) sampling and min-finding.
    """
    write_idx = 0 # Tracks the next position to write data in the tree leaves

    def __init__(self, capacity):
        """
        Initializes the SumTree.

        Args:
            capacity (int): The maximum number of items (leaves) the tree can hold.
        """
        self.capacity = capacity
        # Tree for sums, initialized to 0
        self.sum_tree = np.zeros(2 * capacity - 1)
        # Tree for minimums, initialized to infinity
        # Unused leaves will have infinity, ensuring they don't affect minimums
        self.min_tree = np.full(2 * capacity - 1, float('inf'))
        self.size = 0 # Count of entries currently in the tree

    def _propagate_upwards(self, idx):
        """
        Propagates changes (sum and min) up the tree starting from a leaf node index.

        Args:
            idx (int): The index of the leaf node in the tree array that was updated.
                       (tree_leaf_index = buffer_index + capacity - 1)
        """
        while idx > 0: # Propagate change upwards until the root (idx=0)
            parent = (idx - 1) // 2
            left = 2 * parent + 1
            right = left + 1

            # Ensure sibling index is within bounds (important if capacity is not power of 2)
            if right >= len(self.sum_tree):
                 # This case might occur if capacity isn't a power of 2, handle gracefully
                 # If right child doesn't exist, parent sum/min depends only on left
                 new_sum = self.sum_tree[left]
                 new_min = self.min_tree[left]
            else:
                 new_sum = self.sum_tree[left] + self.sum_tree[right]
                 new_min = min(self.min_tree[left], self.min_tree[right])

            # Update parent's sum and min
            self.sum_tree[parent] = new_sum
            self.min_tree[parent] = new_min

            # Move up to the parent
            idx = parent

    def _retrieve(self, idx, s):
        """Finds the leaf index corresponding to a sample value 's'."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.sum_tree): # Reached a leaf node
            return idx

        if s <= self.sum_tree[left]:
            return self._retrieve(left, s)
        else:
            # Ensure right index exists before accessing
            if right >= len(self.sum_tree):
                 return self._retrieve(left, s) # Should theoretically not happen if s <= total sum
            return self._retrieve(right, s - self.sum_tree[left])

    def total(self):
        """Returns the total priority sum (value of the root node)."""
        return self.sum_tree[0]

    def get_min_priority(self):
        """Returns the minimum priority among all active entries."""
        # The root of the min_tree holds the minimum value of all leaves
        # This works because unused leaves are initialized to infinity
        return self.min_tree[0] if self.size > 0 else 0.0

    def update(self, idx, p):
        """
        Updates the priority 'p' at a given leaf index 'idx'.
        'idx' here is the index in the tree array (capacity-1 + buffer_index).

        Args:
            idx (int): The tree leaf index to update.
            p (float): The new priority value (must be non-negative).
        """
        if not (0 <= idx < len(self.sum_tree)):
             raise IndexError("Index out of tree bounds.")
        if idx < self.capacity - 1:
             raise ValueError("Index must point to a leaf node.")
        if p < 0:
             raise ValueError("Priority must be non-negative.")

        # Update leaf node in both trees
        self.sum_tree[idx] = p
        self.min_tree[idx] = p

        # Propagate changes upwards
        self._propagate_upwards(idx)

    def add(self, p):
        """
        Adds a new priority 'p' to the tree. If the tree is full,
        it overwrites the oldest entry. Handles sum and min tree updates.
        """
        # Get the tree leaf index for the current write position
        idx = self.write_idx + self.capacity - 1

        self.update(idx, p) # Update the tree with the new priority (handles both sum/min)

        self.write_idx = (self.write_idx + 1) % self.capacity # Circle back to the beginning
        self.size = min(self.size + 1, self.capacity) # Increment size, but not beyond capacity

    def get(self, s):
        """
        Gets the leaf index, priority, and corresponding data index
        for a given sample value 's'.
        """
        if self.total() == 0: # Handle empty tree case
             # Cannot sample based on priority, maybe return random or raise error
             # For now, let's return an invalid index/priority to be handled by caller
             print("Warning: Sampling from SumTree with total priority zero.")
             return (0, 0, 0) # Example invalid return

        idx = self._retrieve(0, s) # Get leaf index in the tree array
        dataIdx = idx - self.capacity + 1 # Convert tree leaf index to buffer index
        # Return: leaf index in tree, priority at that leaf, buffer index
        return (idx, self.sum_tree[idx], dataIdx)


# --- Prioritized Replay Buffer Implementation (Using Updated SumTree) ---

class PrioritizedReplayBuffer:
    """
    Fixed-size buffer storing experience tuples (1-D states), sampled based
    on priority using an updated SumTree that tracks minimums.
    """
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, action_size, state_dim, buffer_size, batch_size, seed, device):
        self.action_size = action_size
        self.state_dim = state_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

        # Use the updated SumTree
        self.tree = SumTree(buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


    def _get_priority(self, error):
        """Calculates priority from TD error, ensuring it's positive."""
        # Ensure priority is slightly above zero even for zero error
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        """Adds experience, assigning max priority initially."""
        # Calculate initial priority: Use max existing or 1.0 if empty
        # Note: tree.get_min_priority() gives the min, not max.
        # We still need max for adding new samples initially.
        # Max priority is simply the highest value in the leaf nodes' sum part.
        # A simpler approach often used: just use priority=1.0 for new samples.
        # Or track max priority separately if needed. Let's use 1.0 for simplicity.
        # max_priority = np.max(self.tree.tree[-self.tree.capacity:]) # Find max in leaves
        # if max_priority <= 0: # Handle empty or all-zero case
        #     max_priority = 1.0 # Default max priority

        # Let's calculate the priority based on a default large error upon insertion
        # Or just assign a fixed high value (e.g. 1.0)
        current_max_priority = 1.0 # Default priority for new samples
        # Optional: track actual max priority if needed
        # if self.size > 0:
        #    current_max_priority = np.max(self.tree.tree[self.capacity-1 : self.capacity-1+self.size])
        #    if current_max_priority <= 0: current_max_priority = 1.0


        self.states[self.ptr] = np.float32(state)
        self.actions[self.ptr] = np.int64(action)
        self.rewards[self.ptr] = np.float32(reward)
        self.next_states[self.ptr] = np.float32(next_state)
        self.dones[self.ptr] = np.float32(done)

        # Use the calculated priority for adding to the SumTree
        self.tree.add(current_max_priority)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)


    def sample(self):
        """Samples batch, calculating IS weights using tracked min priority."""
        if self.size == 0:
            raise BufferError("Cannot sample from an empty buffer.")

        current_batch_size = min(self.batch_size, self.size)

        batch_states = np.zeros((current_batch_size, self.state_dim), dtype=np.float32)
        batch_actions = np.zeros((current_batch_size, 1), dtype=np.int64)
        batch_rewards = np.zeros((current_batch_size, 1), dtype=np.float32)
        batch_next_states = np.zeros((current_batch_size, self.state_dim), dtype=np.float32)
        batch_dones = np.zeros((current_batch_size, 1), dtype=np.float32)

        idxs = np.empty((current_batch_size,), dtype=np.int32)
        is_weights = np.empty((current_batch_size, 1), dtype=np.float32)

        total_priority = self.tree.total()
        if total_priority == 0:
             # This shouldn't happen if priorities are always > 0 due to epsilon
             # Fallback: Sample uniformly if total priority is zero
             print("Warning: Total priority is zero. Sampling uniformly.")
             idxs = np.random.choice(self.size, current_batch_size, replace=False)
             is_weights.fill(1.0) # Weights are 1 if sampling uniformly
             # Retrieve data for uniformly sampled indices
             for i, idx in enumerate(idxs):
                  batch_states[i] = self.states[idx]
                  batch_actions[i] = self.actions[idx]
                  batch_rewards[i] = self.rewards[idx]
                  batch_next_states[i] = self.next_states[idx]
                  batch_dones[i] = self.dones[idx]

        else:
             priority_segment = total_priority / current_batch_size
             self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

             # --- MODIFIED SECTION ---
             # Calculate max weight using efficiently tracked minimum priority
             min_priority = self.tree.get_min_priority()
             # Ensure min_prob calculation avoids division by zero
             min_prob = min_priority / total_priority if total_priority > 0 else 0

             # Ensure max_weight calculation avoids division by zero or invalid ops
             if self.size > 0 and min_prob > 0:
                 max_weight = (self.size * min_prob) ** (-self.beta)
             else:
                 max_weight = 1.0 # Fallback or if only one element
             # --- END MODIFIED SECTION ---

             for i in range(current_batch_size):
                 a = priority_segment * i
                 b = priority_segment * (i + 1)
                 s = random.uniform(a, b)

                 (tree_idx, priority, data_idx) = self.tree.get(s)

                 # Calculate Importance Sampling weight
                 sampling_probability = priority / total_priority if total_priority > 0 else 0
                 # Avoid division by zero or power of zero if probability is zero
                 weight = (self.size * sampling_probability) ** (-self.beta) if sampling_probability > 0 else 0

                 is_weights[i, 0] = weight

                 idxs[i] = data_idx
                 batch_states[i] = self.states[data_idx]
                 batch_actions[i] = self.actions[data_idx]
                 batch_rewards[i] = self.rewards[data_idx]
                 batch_next_states[i] = self.next_states[data_idx]
                 batch_dones[i] = self.dones[data_idx]

             # Normalize IS weights only if max_weight is valid and positive
             if max_weight > 0:
                 is_weights /= max_weight
             else:
                  print(f"Warning: max_weight is {max_weight}, cannot normalize IS weights.")
                  is_weights.fill(1.0) # Fallback if normalization fails


        # Convert batch to PyTorch tensors
        states_tensor = torch.from_numpy(batch_states).float().to(self.device)
        actions_tensor = torch.from_numpy(batch_actions).long().to(self.device)
        rewards_tensor = torch.from_numpy(batch_rewards).float().to(self.device)
        next_states_tensor = torch.from_numpy(batch_next_states).float().to(self.device)
        dones_tensor = torch.from_numpy(batch_dones).float().to(self.device)
        is_weights_tensor = torch.from_numpy(is_weights).float().to(self.device)

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
             if not (0 <= idx < self.buffer_size):
                 print(f"Warning: Attempting to update priority for invalid buffer index {idx}. Skipping.")
                 continue
             # Note: Index check against self.size is not strictly needed here
             # because batch_indices come from `sample`, which only returns valid indices.

             priority = self._get_priority(error)
             # Clamp priority to avoid potential numerical issues? Optional.
             # priority = np.max([priority, 1e-6]) # Ensure priority is at least small positive

             tree_idx = idx + self.tree.capacity - 1
             self.tree.update(tree_idx, priority) # Use the updated SumTree method

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size

# --- Example Usage (Modified to show min priority) ---
if __name__ == '__main__':
    STATE_DIM = 10
    BUFFER_SIZE = 1000 # Smaller buffer for quicker testing
    BATCH_SIZE = 32
    ACTION_SIZE = 4
    SEED = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    buffer = PrioritizedReplayBuffer(ACTION_SIZE, STATE_DIM, BUFFER_SIZE, BATCH_SIZE, SEED, DEVICE)

    def create_dummy_experience(state_dim, action_size):
        state = np.random.randn(state_dim).astype(np.float32)
        action = random.randint(0, action_size - 1)
        reward = random.uniform(-1, 1)
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = random.choice([True, False])
        return state, action, reward, next_state, done

    print("Adding experiences...")
    num_to_add = BUFFER_SIZE // 2
    for i in range(num_to_add):
        exp = create_dummy_experience(STATE_DIM, ACTION_SIZE)
        buffer.add(*exp) # Adds with default priority 1.0

    print(f"Buffer size: {len(buffer)}")
    print(f"SumTree total priority: {buffer.tree.total():.2f}")
    print(f"SumTree min priority (initial): {buffer.tree.get_min_priority():.6f}") # Should be 1.0

    if len(buffer) >= BATCH_SIZE:
        print("\nSampling a batch...")
        experiences, idxs, is_weights = buffer.sample()
        print(f"Sampled indices: {idxs[:5]}...")
        print(f"Sampled IS weights: {is_weights[:5].squeeze().cpu().numpy()}...")

        print("\nUpdating priorities with varying errors...")
        # Generate errors, some small, some large
        errors = np.concatenate([
            np.random.rand(BATCH_SIZE // 2) * 0.1, # Small errors
            np.random.rand(BATCH_SIZE - BATCH_SIZE // 2) * 10 # Large errors
        ])
        np.random.shuffle(errors)
        print(f"Dummy TD Errors (first 5): {errors[:5]}")

        buffer.update_priorities(idxs, errors)
        print("Priorities updated.")
        print(f"SumTree total priority after update: {buffer.tree.total():.2f}")
        # Check the new minimum priority - should reflect the smallest calculated priority
        new_min_p = buffer.tree.get_min_priority()
        print(f"SumTree min priority after update: {new_min_p:.6f}")

        # Verify calculation: smallest priority should be approx ((min_error + epsilon)^alpha)
        min_calc_p = (np.min(np.abs(errors)) + buffer.epsilon) ** buffer.alpha
        print(f"Expected min priority (approx):   {min_calc_p:.6f}")

        print("\nSampling another batch...")
        experiences2, idxs2, is_weights2 = buffer.sample()
        print(f"Sampled indices (second batch): {idxs2[:5]}...")
        print(f"Buffer beta value after sampling: {buffer.beta:.4f}")

    else:
        print("Buffer not full enough to sample.")