def attention_step(q, k_new, v_new, cache=None):
    """Append projected rows and return outputs, weights, and the new cache."""
    q, k_new, v_new = map(np.asarray, (q, k_new, v_new))
    if any(a.ndim != 4 for a in (q, k_new, v_new)):
        raise ValueError("q, k_new, and v_new need four axes")
    query_length = q.shape[-2]
    if not query_length == k_new.shape[-2] == v_new.shape[-2] > 0:
        raise ValueError("each query needs one new key row and value row")

    past_length = 0
    if cache is None:
        keys, values = k_new.copy(), v_new.copy()
    else:
        old_keys, old_values = cache
        past_length = old_keys.shape[-2]
        if old_values.shape[-2] != past_length:
            raise ValueError("cached key and value histories must agree")
        if old_keys.dtype != k_new.dtype or old_values.dtype != v_new.dtype:
            raise ValueError("cache dtype must stay fixed")
        keys = np.concatenate((old_keys, k_new), axis=-2)
        values = np.concatenate((old_values, v_new), axis=-2)

    allowed = causal_mask(past_length, query_length, keys.shape[-2])
    output, weights = grouped_attention(q, keys, values, allowed)
    return output, weights, (keys, values)
