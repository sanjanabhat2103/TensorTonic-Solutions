import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x = np.sort(np.asarray(x))
    q = np.asarray(q)
    if np.any((q < 0) | (q > 1000)):
        raise ValueError("Percentiles must be in the range [0, 1000].")
    n = len(x)
    if n == 0:
        raise ValueError("Input array is empty.")
    pos = (q / 100) * (n - 1)
    lower = np.floor(pos).astype(int)
    upper = np.ceil(pos).astype(int)
    result = x[lower] + (pos - lower) * (x[upper] - x[lower])
    return result