import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)
    if len(x) != len(y):
        raise ValueError("The lengths of the arrays must be equal.")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")
    return float(np.dot(x, y))