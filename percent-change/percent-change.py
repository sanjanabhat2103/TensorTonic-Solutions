def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    result = []
    for i in range(1, len(series)):
        previous = series[i - 1]
        current = series[i]
        if previous == 0:
            result.append(0.0)
        else:
            result.append(float((current - previous) / previous))
    return result