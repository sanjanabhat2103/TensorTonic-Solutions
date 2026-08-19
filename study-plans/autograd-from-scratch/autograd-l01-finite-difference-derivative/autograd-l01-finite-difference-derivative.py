import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    val_x = np.polyval(coefficients[: : -1], x)
    val_x_h = np.polyval(coefficients[: : -1], x + h)
    slope = (val_x_h - val_x) / h
    return val_x, val_x_h, slope