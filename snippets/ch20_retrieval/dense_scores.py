def dense_scores(query_vector, passage_vectors):
    """Compute raw dot products in the supplied passage order."""
    query = _real_array(query_vector, "query vector")
    passages = _real_array(passage_vectors, "passage vectors")
    if (query.ndim != 1 or query.size == 0 or passages.ndim != 2
            or passages.shape[1] != query.size):
        raise ValueError("use a query vector and a passage matrix of the same width")
    if not (np.isfinite(query).all() and np.isfinite(passages).all()):
        raise ValueError("query and passage vectors must be finite")
    with np.errstate(over="raise", invalid="raise"):
        return passages @ query
