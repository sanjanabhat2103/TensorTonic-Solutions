import numpy as np
from math import exp, factorial

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.

    Parameters:
        lam : float
            Mean (λ) of the Poisson distribution.
        k : int
            Number of events.

    Returns:
        (pmf, cdf)
    """
    if lam < 0 or k < 0:
        return None
    pmf = (lam ** k) * exp(-lam) / factorial(k)
    cdf = sum((lam ** i) * exp(-lam) / factorial(i) for i in range(k + 1))
    return pmf, cdf