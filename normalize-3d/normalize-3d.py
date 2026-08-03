import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.asarray(v, dtype = float)
    if v.shape[-1] != 3:
        raise ValueError("Input must contain 3D vectors.")
    norm = np.sqrt(np.sum(v ** 2, axis = -1, keepdims = True))
    return np.divide(v, norm, out = np.zeros_like(v), where = norm != 0)