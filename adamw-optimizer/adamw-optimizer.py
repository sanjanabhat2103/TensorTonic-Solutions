import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    w = np.asarray(w, dtype = float)
    m = np.asarray(m, dtype = float)
    v = np.asarray(v, dtype = float)
    grad = np.asarray(grad, dtype = float)
    if beta1 <= 0 or beta1 >= 1:
        raise ValueError("Beta1 must lie between 0 and 1.")
    if beta2 <= 0 or beta2 >= 1:
        raise ValueError("Beta2 must lie between 0 and 1.")
    if lr <= 0:
        raise ValueError("Learning rate must be positive.")
    if weight_decay < 0:
        raise ValueError("Weight decay must be non-negative.")
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    w = w - lr * (weight_decay * w) - lr * m / ((np.sqrt(v) + eps))
    return w, m, v