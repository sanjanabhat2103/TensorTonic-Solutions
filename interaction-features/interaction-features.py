import numpy as np

def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append((X[: , i] * X[: , j]).reshape(-1, 1))
    if interactions:
        X_new = np.hstack([X] + interactions)
    else:
        X_new = X
    return X_new.tolist()
