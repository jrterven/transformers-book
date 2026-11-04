def embedding_lookup(embedding_table, token_ids):
    """Append the embedding-width axis to the token-ID array's shape."""
    embedding_table = np.asarray(embedding_table, dtype=np.float64)
    if embedding_table.ndim != 2 or min(embedding_table.shape) < 1:
        raise ValueError("embedding_table must be a nonempty [V, d] matrix")
    token_ids = _check_ids(token_ids, embedding_table.shape[0])
    return embedding_table[token_ids]
