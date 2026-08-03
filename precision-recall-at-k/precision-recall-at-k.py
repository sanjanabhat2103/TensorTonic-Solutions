def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    recommended_set = set(recommended_k)
    hits = len(recommended_set & relevant_set)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0
    return [float(precision), float(recall)]