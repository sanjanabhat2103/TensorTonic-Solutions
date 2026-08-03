import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    k = np.asarray(k, dtype=float)
    if p <= 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    if np.any(k < 1):
        raise ValueError("k must be >= 1")
    pmf = ((1 - p) ** (k - 1)) * p
    mean = 1 / p
    return pmf, float(mean)