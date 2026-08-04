def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    if alpha <= 0 or alpha > 1:
        return []
    if len(values) == 0:
        return []
    ema = []
    ema.append(values[0])
    for i in range(1, len(values)):
        current_ema = alpha * values[i] + (1 - alpha) * ema[i - 1]
        ema.append(current_ema)
    return ema
        