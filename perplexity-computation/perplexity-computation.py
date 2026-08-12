def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    probs = np.asarray(prob_distributions)
    actual_tokens = np.asarray(actual_tokens)
    token_probs = probs[np.arange(len(actual_tokens)), actual_tokens]
    return float(np.exp(-np.mean(np.log(token_probs))))
    