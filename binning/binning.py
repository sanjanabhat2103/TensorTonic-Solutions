import math 

def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    if len(values) == 0:
        return []
    if num_bins < 1:
        return []
    w = (max(values) - min(values)) / (num_bins)
    if w == 0:
        return [0] * len(values)
    bin = []
    for i in range(len(values)):
        bin.append(min(num_bins - 1, math.floor((values[i] - min(values)) / w)))
    return bin