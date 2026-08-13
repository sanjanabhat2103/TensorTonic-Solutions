import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    w = np.asarray(w)
    grad = np.asarray(grad)
    E_grad_sq = np.asarray(E_grad_sq)
    E_update_sq = np.asarray(E_update_sq)
    E_grad_sq = rho * E_grad_sq + (1 - rho) * grad ** 2
    w_change = -(np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq + eps)) * grad
    E_update_sq = rho * E_update_sq + (1 - rho) * w_change ** 2
    w = w + w_change
    return w, E_grad_sq, E_update_sq