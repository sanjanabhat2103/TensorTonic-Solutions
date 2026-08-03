import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    v = np.asarray(v, dtype = float)
    if v.shape[-1] != 3:
        raise ValueError("Input must contain 3D vectors.")
    return np.sqrt(np.sum(v ** 2, axis = -1))