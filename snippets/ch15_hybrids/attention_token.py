def attention_token(query, key, value, cache):
    """Append one position to a full-attention cache and read the whole prefix."""
    query, key, value, old_keys, old_values = _attention_inputs(
        query, key, value, cache
    )
    keys = np.concatenate((old_keys, key[None, :]), axis=0)
    values = np.concatenate((old_values, value[None, :]), axis=0)
    output, _ = scaled_dot_product_attention(query[None, :], keys, values)
    return output[0], (keys, values)
