import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    W_hh = np.asarray(W_hh, dtype = float)
    spectral_norm = np.linalg.norm(W_hh, ord = 2)
    norms = []
    for t in range(T):
        norms.append(float(spectral_norm ** t))
    return norms