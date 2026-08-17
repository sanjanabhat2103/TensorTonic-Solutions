import numpy as np

def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    series = np.asarray(series, dtype = float)
    n = len(series)
    if n == 0:
        return np.array([], dtype = float)
    levels = np.zeros(n)
    trends = np.zeros(n)
    levels[0] = series[0]
    if n > 1:
        trends[0] = series[1] - series[0]
    else:
        trends[0] = 0.0
    for t in range(1, n):
        levels[t] = alpha * series[t] + (1 - alpha) * (levels[t - 1] + trends[t - 1])
        trends[t] = beta * (levels[t] - levels[t - 1]) + (1 - beta) * trends[t - 1]
    return levels.tolist()