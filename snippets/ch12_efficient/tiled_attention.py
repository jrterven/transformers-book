def tiled_attention(q, k, v, query_tile, key_tile, *, causal=False, query_start=0):
    """Return one output row per query, using compact score tiles."""
    q, k, v, query_tile, key_tile, query_start = _tiled_inputs(
        q, k, v, query_tile, key_tile, causal, query_start
    )
    output = np.empty((len(q), v.shape[1]))
    scale = math.sqrt(q.shape[1])
    for q_start in range(0, len(q), query_tile):
        queries = q[q_start:q_start + query_tile]
        positions = query_start + q_start + np.arange(len(queries))
        maximum = np.full(len(queries), -np.inf)
        denominator = np.zeros(len(queries))
        numerator = np.zeros((len(queries), v.shape[1]))
        for k_start in range(0, len(k), key_tile):
            keys = k[k_start:k_start + key_tile]
            values = v[k_start:k_start + key_tile]
            allowed = None
            if causal:
                key_positions = k_start + np.arange(len(keys))
                allowed = key_positions[None, :] <= positions[:, None]
                if not allowed.any():
                    continue
            scores = (queries @ keys.T) / scale
            maximum, denominator, numerator = online_update(
                maximum, denominator, numerator, scores, values, allowed
            )
        output[q_start:q_start + len(queries)] = numerator / denominator[:, None]
    return output
