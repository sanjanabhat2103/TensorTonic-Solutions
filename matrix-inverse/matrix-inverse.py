import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    A = np.asarray(A, dtype = float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    if np.isclose(0, np.linalg.det(A)):
        return None
    A_inv = np.linalg.inv(A)
    return A_inv
        