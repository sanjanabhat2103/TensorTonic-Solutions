import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    
    Uses first-visit Monte Carlo policy evaluation.
    """
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)
    for episode in episodes:
        G = 0.0
        visited = set()
        returns = []
        for state, reward in reversed(episode):
            G = reward + gamma * G
            returns.append((state, G))
        for state, G in reversed(returns):
            if state not in visited:
                visited.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
    V = np.divide(
        returns_sum,
        returns_count,
        out = np.zeros(n_states),
        where = returns_count != 0
    )
    return V
