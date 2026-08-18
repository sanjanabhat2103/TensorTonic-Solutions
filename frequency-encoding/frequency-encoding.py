def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    n = len(values)
    freq = []
    for i in values:
        freq.append(values.count(i) / n)
    return freq