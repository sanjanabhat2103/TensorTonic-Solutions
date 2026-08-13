def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    assignments = []
    for point in points:
        distances = [
            sum((p - c) ** 2 for p, c in zip(point, centroid))
            for centroid in centroids
        ]
        assignments.append(distances.index(min(distances)))
    return assignments