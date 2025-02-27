import numpy as np
from collections import defaultdict

def _policy_s(Q_s, epsilon, num_actions):
    # Probability of selecting any action:
    probs = np.ones(num_actions) * (epsilon / num_actions)
    # Determine which is the greedy action and augment it's probability
    best_action = np.argmax(Q_s)
    probs[best_action] += 1 - epsilon
    # Ensure the total probability sums to 1
    assert abs(np.sum(probs) - 1) < 1e-8
    return probs

def _choose_action(nA, Q_s, epsilon):
    """Choose an action given an epsilon-greedy policy"""
    action_probability = _policy_s(Q_s, epsilon, nA)

    return np.random.choice(np.arange(nA), p=action_probability)

def _sarsa_update_Q(Q_t, reward, Q_tp1, alpha, gamma):
    """Update the Q-table using a Sarsa update strategy"""
    """The type of sarsa will be dictated by the value of Q_tp1"""
    return Q_t + alpha * ((reward + gamma * Q_tp1) - Q_t)

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self._eps = 0.005
        self._alpha = 0.95
        self._gamma = 1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return _choose_action(self.nA, self.Q[state], self._eps)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        Q_t = self.Q[state][action]

        # Implemented the Expected Sarsa value update method
        probs = _policy_s(self.Q[next_state], self._eps, self.nA)
        Q_tp1 = np.dot(self.Q[next_state], probs)

        self.Q[state][action] = _sarsa_update_Q(Q_t, reward, Q_tp1, self._alpha, self._gamma)

        # Decay epsilon in order to stabilize learning
        self._eps *= 0.9995
