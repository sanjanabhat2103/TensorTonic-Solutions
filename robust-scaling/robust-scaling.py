import numpy as np

def robust_scaling(values):
    """
    Scale values using median and interquartile range (IQR).
    """
    values = np.asarray(values, dtype = float)
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    median = np.median(sorted_vals)
    if n == 1:
        return [0]
    if n % 2 == 0:
        lower = sorted_vals[: n // 2]
        upper = sorted_vals[n // 2: ]
    else:
        lower = sorted_vals[: n // 2]
        upper = sorted_vals[n // 2 + 1: ]
    q1 = np.median(lower) if len(lower) else median
    q3 = np.median(upper) if len(upper) else median
    iqr = q3 - q1
    if iqr == 0:
        return (values - median).tolist()
    return ((values - median) / iqr).tolist()