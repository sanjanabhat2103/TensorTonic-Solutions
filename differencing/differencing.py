def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    diff_series = series.copy()
    for i in range(order):
        diff_series = [diff_series[j] - diff_series[j - 1] for j in range(1, len(diff_series))]
    return diff_series