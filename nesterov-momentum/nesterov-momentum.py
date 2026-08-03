import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    w = np.asarray(w, dtype = float)
    v = np.asarray(v, dtype = float)
    grad = np.asarray(grad, dtype = float)
    if len(w) != len(v) or len(v) != len(grad) or len(w) != len(grad):
        raise ValueError("Shapes of w, v and grad must match.")
    if lr <= 0:
        raise ValueError("Learning rate must be positive.")
    if momentum < 0 or momentum >= 1:
        raise ValueError("Momentum must lie between 0 and 1.")
    w_look = w - momentum * v
    v = momentum * v + lr * grad
    w = w - v
    return w, v
    
    