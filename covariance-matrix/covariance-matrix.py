import numpy as np

def covariance_matrix(X):
    """
    Compute the covariance matrix of dataset X.

    Parameters:
        X : array-like of shape (n_samples, n_features)

    Returns:
        Covariance matrix of shape (n_features, n_features),
        or None if the input is invalid.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    X_centered = X - np.mean(X, axis=0)
    n_samples = X.shape[0]
    cov = (X_centered.T @ X_centered) / (n_samples - 1)
    return cov