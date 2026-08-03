import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.asarray(x, dtype = float)
    mean = np.mean(x)
    median = np.median(x)
    counts = Counter(x)
    mode = max(counts, key = counts.get)
    return float(mean), float(median), float(mode)