import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    s = np.sqrt(np.sum((x - mean) ** 2) / (n - 1))
    return (mean - mu0) / (s / np.sqrt(n))