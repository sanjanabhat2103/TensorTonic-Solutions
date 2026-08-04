import numpy as np 

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = np.asarray(x1, dtype = float)
    x2 = np.asarray(x2, dtype = float)
    if len(x1) != len(x2):
        return 0.0
    if len(x1) < 1 or len(x2) < 1:
        return 0.0
    norm_1 = np.linalg.norm(x1)
    norm_2 = np.linalg.norm(x2)
    if norm_1 <= 0 or norm_2 <= 0:
        return 0.0
    if label != 1 and label != -1:
        return 0.0
    if margin < 0:
        return 0.0
    cos_sim = np.dot(x1, x2) / (norm_1 * norm_2)
    if label == 1:
        return 1 - cos_sim
    elif label == -1:
        return max(0, cos_sim - margin)