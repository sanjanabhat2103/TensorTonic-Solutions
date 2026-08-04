import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    Z1 = np.asarray(Z1, dtype = float)
    Z2 = np.asarray(Z2, dtype = float)
    S = np.dot(Z1, Z2.T) / temperature
    S = S - np.max(S, axis = 1, keepdims = True)
    exp_S = np.exp(S)
    positives = np.diag(exp_S)
    denominators = np.sum(exp_S, axis = 1)
    loss = -np.mean(np.log(positives / denominators))
    return float(loss)