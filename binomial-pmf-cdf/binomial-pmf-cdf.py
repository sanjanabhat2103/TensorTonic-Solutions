import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    if k < 0 or k > n:
        raise ValueError("k must be between 0 and n")
    pmf = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    cdf = 0
    for i in range(k + 1):
        cdf += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return float(pmf), float(cdf)