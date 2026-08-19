import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    """
    Returns: updated values and the predicted first-order objective change
    """
    values = np.asarray(values, dtype = float)
    gradients = np.asarray(gradients, dtype = float)
    values -= learning_rate * gradients
    return (values.tolist(), np.sum(-learning_rate * gradients ** 2))