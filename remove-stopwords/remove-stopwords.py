def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    new_l = []
    for i in tokens:
        if i not in stopwords:
            new_l.append(i)
    return new_l