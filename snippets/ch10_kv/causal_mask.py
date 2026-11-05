def causal_mask(query_start, query_length, key_length):
    """True if a query can read the key, with positions starting at zero."""
    query_start, query_length, key_length = map(
        index, (query_start, query_length, key_length)
    )
    if not 0 <= query_start < query_start + query_length <= key_length:
        raise ValueError("queries must fit inside the key prefix")
    query_positions = query_start + np.arange(query_length)
    key_positions = np.arange(key_length)
    return key_positions[None, :] <= query_positions[:, None]
