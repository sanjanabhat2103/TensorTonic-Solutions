import numpy as np

def calculate_eigenvalues(matrix):
    try:
        matrix = np.asarray(matrix, dtype = complex)
    except ValueError:
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    eigs = np.linalg.eigvals(matrix)
    eigs = np.round(eigs, 12)
    eigs = np.array(sorted(eigs, key = lambda x: (x.real, x.imag)))
    return eigs