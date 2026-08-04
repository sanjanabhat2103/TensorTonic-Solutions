import numpy as np

def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    n = len(values)
    if n < 1:
        return []
    if degree < 0:
        return []
    powers = []
    for i in range(degree + 1):
        powers.append(np.power(values, i))
    return np.column_stack(powers).tolist()