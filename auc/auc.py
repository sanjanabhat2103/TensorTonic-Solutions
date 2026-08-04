import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.asarray(fpr, dtype = float)
    tpr = np.asarray(tpr, dtype = float)
    if len(fpr) != len(tpr) or len(fpr) < 2:
        return 0.0
    return np.sum((fpr[1: ] - fpr[: -1]) * (tpr[1: ] + tpr[: -1]) / 2)
