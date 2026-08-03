import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def train_logistic_regression(X, y, lr = 0.1, steps = 1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = float)
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    for i in range(steps):
        p = _sigmoid(X @ w + b)
        w -= lr * (X.T @ (p - y)) / n_samples
        b -= lr * np.mean(p - y)
    return w, b