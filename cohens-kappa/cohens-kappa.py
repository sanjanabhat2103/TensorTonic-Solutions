import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)
    po = np.mean(rater1 == rater2)
    labels = np.union1d(rater1, rater2)
    pe = sum(
        np.mean(rater1 == label) * np.mean(rater2 == label)
        for label in labels
    )
    if np.isclose(1 - pe, 0):
        return 1.0 if np.isclose(po, 1) else 0.0
    return (po - pe) / (1 - pe)