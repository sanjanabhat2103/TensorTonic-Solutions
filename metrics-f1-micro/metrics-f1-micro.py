import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.asarray(y_true, dtype = int)
    y_pred = np.asarray(y_pred, dtype = int)
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of the arrays should be equal.")
    return np.mean(y_true == y_pred)