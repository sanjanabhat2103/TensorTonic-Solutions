import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    vocab_map = {word: idx for idx, word in enumerate(vocab)}
    vector = np.zeros(len(vocab), dtype = int)
    for token in tokens:
        if token in vocab_map:
            vector[vocab_map[token]] += 1 
    return vector