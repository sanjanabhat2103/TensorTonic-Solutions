def rank_transform(values):
    """
    Replace each value with its average rank.
    Equal values receive the average of their ranks.
    """
    n = len(values)
    sorted_vals = sorted((value, index) for index, value in enumerate(values))
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j < n and sorted_vals[j][0] == sorted_vals[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[sorted_vals[k][1]] = avg_rank
        i = j
    return ranks
