import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    if len(v) != 3 or len(w) != 3:
        raise ValueError("Input vectors must be 3 dimensional.")
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm == 0 or w_norm == 0:
        return np.nan
    cosine = np.dot(v, w) / (v_norm * w_norm)
    if cosine < -1 or cosine > 1:
        raise ValueError("Cosine must lie between -1 and 1.")
    return np.arccos(cosine)