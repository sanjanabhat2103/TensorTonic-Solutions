import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    n_l = len(y_left)
    n_r = len(y_right)
    n = n_l + n_r
    if n == 0:
        return 0.0
    def gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts = True)
        probs = counts / len(y)
        return float(1.0 - np.sum(probs ** 2))
    return float(
        (n_l * gini(y_left) + n_r * gini(y_right)) / n
    )