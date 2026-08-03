import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    r, c = A.shape
    transpose = np.zeros((c, r), dtype = A.dtype)
    for i in range(r):
        for j in range(c):
            transpose[j][i] = A[i][j]
    return transpose