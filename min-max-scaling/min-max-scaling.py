import numpy as np

def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    data = np.asarray(data, dtype = float)
    min_col = np.min(data, axis = 0)
    max_col = np.max(data, axis = 0)
    denominator = max_col - min_col
    scaled = np.zeros_like(data, dtype = float)
    mask = denominator != 0
    scaled[:, mask] = (data[:, mask] - min_col[mask]) / denominator[mask]
    return scaled.tolist()