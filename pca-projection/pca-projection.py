import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.

    Args:
        X: NumPy array of shape (n_samples, n_features)
        k: Number of principal components

    Returns:
        Projected data of shape (n_samples, k)
    """
    X_centred = X - np.mean(X, axis = 0)
    cov_matrix = np.cov(X_centred, rowvar = False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[: : -1]
    principal_components = eigenvectors[:, idx[: k]]
    X_projected = X_centred @ principal_components
    return X_projected