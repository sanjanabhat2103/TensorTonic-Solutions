def word_count_dict(sentences):
    """
    sentences: list of lists of words
    Returns: dict[str, int] - global word frequency
    """
    counts = {}
    for sentence in sentences:
        for word in sentence:
            counts[word] = counts.get(word, 0) + 1
    return counts