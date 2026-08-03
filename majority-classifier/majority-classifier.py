import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    counts = {}
    for label in y_train:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    majority = max(counts, key = counts.get)
    return np.array([majority] * len(X_test))