import numpy as np

def expected_value_discrete(x, p):
    x = np.asarray(x, dtype = float)
    p = np.asarray(p, dtype = float)
    if len(x) != len(p):
        raise ValueError("x and p should be of the same length.")
    if not np.isclose(np.sum(p), 1):
        raise ValueError("The sum of the probabilities must be 1.")
    if np.any(p < 0):
        raise ValueError("Probabilities cannot be negative.")
    return np.sum(x * p)