def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    if window_size < 1 or window_size > len(values):
        return []
    sma = []
    for i in range(len(values) - window_size + 1):
        window = values[i: i + window_size]
        sma.append(sum(window) / window_size)
    return sma