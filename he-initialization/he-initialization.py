import numpy as np 

def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    fan_in = float(fan_in)
    W = np.asarray(W, dtype = float)
    l = np.sqrt(6 / fan_in)
    w_new = W * 2 * l - l
    return w_new