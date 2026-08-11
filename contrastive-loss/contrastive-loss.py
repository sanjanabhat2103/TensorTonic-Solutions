import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a = np.asarray(a, dtype = float)
    b = np.asarray(b, dtype = float)
    y = np.asarray(y, dtype = float)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must have shape (N, D) or (D,)")

    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must have the same feature dimension")
    try:
        a, b = np.broadcast_arrays(a, b)
    except ValueError:
        raise ValueError("a and b must be broadcastable to the same shape")
    y = np.ravel(y)
    if y.shape[0] != a.shape[0]:
        raise ValueError("y must have one label per pair")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")
    if margin <= 0:
        raise ValueError("margin must be > 0")
    if reduction not in ("mean", "sum"):
        raise ValueError('reduction must be "mean" or "sum"')
    dist_sq = np.sum((a - b) ** 2, axis=1)
    loss = y * dist_sq + (1 - y) * np.maximum(margin - np.sqrt(dist_sq), 0.0) ** 2
    if reduction == "mean":
        return float(np.mean(loss))
    return float(np.sum(loss))