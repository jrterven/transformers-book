def image_feature_bridge(queries, keys, values, projection):
    """Read image features with queries and project the result to LM width."""
    queries = _real_matrix(queries, "queries")
    keys = _real_matrix(keys, "keys")
    values = _real_matrix(values, "values")
    projection = _real_matrix(projection, "projection")
    if projection.shape[0] != values.shape[1]:
        raise ValueError("projection rows must match the value width")
    context, weights = scaled_dot_product_attention(queries, keys, values)
    with np.errstate(over="raise", invalid="raise"):
        language_tokens = context @ projection.astype(np.float64, copy=False)
    return context, weights, language_tokens
