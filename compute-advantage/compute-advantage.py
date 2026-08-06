import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    states = np.asarray(states)
    rewards = np.asarray(rewards, dtype = float)
    V = np.asarray(V, dtype = float)
    n = len(rewards)
    A = np.zeros(n, dtype = float)
    G = 0.0
    for t in range(n - 1, -1, -1):
        G = rewards[t] + gamma * G
        A[t] = G - V[states[t]]
    return A