import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.

    Parameters:
        y_true (numpy.ndarray): True class labels, shape (n_samples,)
        y_pred (numpy.ndarray): Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        float: Average cross-entropy loss
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype = float)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n_samples = y_true.shape[0]
    loss = -np.mean(np.log(y_pred[np.arange(n_samples), y_true]))
    return loss