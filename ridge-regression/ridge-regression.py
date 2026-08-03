import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = float)
    n_features = X.shape[1]
    I = np.eye(n_features)
    return np.linalg.solve(X.T @ X + lam * I, X.T @ y)