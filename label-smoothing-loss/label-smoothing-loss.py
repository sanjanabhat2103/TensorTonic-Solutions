import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    predictions = np.asarray(predictions)
    q = []
    for i in range(len(predictions)):
        if i == target:
            q.append((1 - epsilon) + epsilon / len(predictions))
        else:
            q.append(epsilon / len(predictions))
    q = np.asarray(q, dtype = float)
    return -np.sum(q * np.log(predictions))