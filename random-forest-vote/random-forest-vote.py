import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    predictions = np.asarray(predictions)
    maj = []
    for i in range(predictions.shape[1]):
        values, counts = np.unique(predictions[:, i], return_counts = True)
        maj.append(values[np.argmax(counts)])
    return maj