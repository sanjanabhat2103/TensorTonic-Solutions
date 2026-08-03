import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.asarray(x, dtype = float)
    n = len(x)
    if n < 2:
        raise ValueError("At least 2 samples are required.")
    mean = np.mean(x)
    sam_var = np.sum((x - mean) ** 2) / (n - 1)
    sam_std = np.sqrt(sam_var)
    return sam_var, sam_std