import numpy as np

def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    values = np.asarray(values)
    theta = 2 * np.pi * values / period
    return np.column_stack((np.sin(theta), np.cos(theta))).tolist()