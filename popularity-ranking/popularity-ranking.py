def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    scores = []
    for item in items:
        R = item[0]
        v = item[1]
        score = (v / (v + min_votes)) * R + (min_votes / (v + min_votes)) * global_mean
        scores.append(score)
    return scores