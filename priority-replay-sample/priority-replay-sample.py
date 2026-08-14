import numpy as np 

def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    priorities = np.asarray(priorities, dtype = float)
    p = priorities ** alpha / np.sum(priorities ** alpha)
    N = len(priorities)
    w = 1 / (N * p) ** beta
    W = w / np.max(w)
    return [p.tolist(), W.tolist()]