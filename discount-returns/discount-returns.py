import numpy as np

def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    if gamma < 0 or gamma > 1:
        return []
    rewards = np.asarray(rewards, dtype = float)
    G = np.zeros_like(rewards)
    running_return = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running_return = rewards[i] + gamma * running_return
        G[i] = running_return
    return G
