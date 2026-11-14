def clipped_ngram_precision(candidate, reference, n):
    """Return clipped precision against one reference, or None for zero trials."""
    n = index(n)
    if n < 1:
        raise ValueError("n must be positive")
    candidate, reference = _tokens(candidate), _tokens(reference)
    count = len(candidate) - n + 1
    if count <= 0:
        return None
    actual = _ngram_counts(candidate, n)
    allowed = _ngram_counts(reference, n)
    matched = sum(min(number, allowed[gram]) for gram, number in actual.items())
    return matched / count
