import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    x = np.asarray(x, dtype = float)
    if p < 0 or p > 1:
        raise ValueError("Probabilities must be from 0 to 1")
    pmf = np.where(x == 1, p, 1 - p)
    mean = p 
    var = p * (1 - p)
    return pmf, mean, var