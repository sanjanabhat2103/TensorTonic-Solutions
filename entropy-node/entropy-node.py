import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y, dtype = float)
    if len(y) == 0:
        return 0.0
    p = np.unique(y, return_counts = True)[1] / len(y)
    return -np.sum(p * np.log2(p))