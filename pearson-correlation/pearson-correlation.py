import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    X = np.asarray(X, dtype = float)
    X_centred = X - np.mean(X, axis = 0)
    cov = np.dot(X_centred.T, X_centred) / (X.shape[0] - 1)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return corr