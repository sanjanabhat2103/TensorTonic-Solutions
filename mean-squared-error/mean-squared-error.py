import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred = np.asarray(y_pred, dtype = float)
    y_true = np.asarray(y_true, dtype = float)
    if len(y_pred) != len(y_true):
        return None
    n = len(y_pred)
    return np.sum((y_pred - y_true) ** 2) / n
