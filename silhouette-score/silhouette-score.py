import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    n = len(X)
    scores = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = np.mean(np.linalg.norm(X[i] - X[same], axis = 1)) if np.any(same) else 0
        b = min(np.mean(np.linalg.norm(X[i] - X[labels == c], axis = 1)) for c in np.unique(labels) if c != labels[i])
        scores.append((b - a) / max(a, b))
    return np.mean(scores)