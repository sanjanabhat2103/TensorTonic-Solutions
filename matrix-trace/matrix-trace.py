import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A = np.asarray(A, dtype = float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("The matrix must be square.")
    trace = 0.0
    n = A.shape[0]
    for i in range(n):
        trace += A[i, i]
    return trace
    