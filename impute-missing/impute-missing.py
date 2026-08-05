import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values using column mean or median.
    """
    X = np.asarray(X, dtype = float).copy()
    if X.ndim == 1:
        if np.all(np.isnan(X)):
            return np.zeros_like(X)
        if strategy == 'mean':
            value = np.nanmean(X)
        elif strategy == 'median':
            value = np.nanmedian(X)
        else:
            raise ValueError("Imputation strategy must be 'mean' or 'median'")
        X[np.isnan(X)] = value
        return X
    for col in range(X.shape[1]):
        if strategy == 'mean':
            value = np.nanmean(X[: , col])
        elif strategy == 'median':
            value = np.nanmedian(X[: , col])
        else:
            raise ValueError("Imputation strategy must be 'mean' or 'median'")
        if np.isnan(value):
            value = 0.0
        X[np.isnan(X[: , col]), col] = value
    return X