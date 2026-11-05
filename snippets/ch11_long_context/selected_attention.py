def selected_attention(query, keys, values, index_scores, k, allowed=None):
    """Return one output, the selected positions, and their main weights."""
    query, keys, values = map(np.asarray, (query, keys, values))
    index_scores = np.asarray(index_scores, dtype=np.float64)
    if query.ndim != 1 or keys.ndim != 2 or values.ndim != 2:
        raise ValueError("need one query and matrices of key and value rows")
    if keys.shape != (len(values), len(query)):
        raise ValueError("query width or key and value row counts disagree")
    if index_scores.shape != (len(keys),):
        raise ValueError("need one index score per key")
    k = index(k)
    if k < 1:
        raise ValueError("k must be positive")

    if allowed is None:
        allowed = np.ones(len(keys), dtype=bool)
    allowed = np.asarray(allowed)
    if allowed.shape != index_scores.shape or allowed.dtype != np.bool_:
        raise ValueError("allowed must be a boolean vector over key positions")
    candidates = np.flatnonzero(allowed)
    scores = index_scores[candidates]
    if len(candidates) == 0 or not np.isfinite(scores).all():
        raise ValueError("need at least one candidate and finite candidate scores")

    # Stable sorting preserves earlier positions when index scores are tied.
    order = np.argsort(-scores, kind="stable")
    selected = candidates[order[:k]]
    output, weights = scaled_dot_product_attention(
        query[None, :], keys[selected], values[selected]
    )
    return output[0], selected, weights[0]
