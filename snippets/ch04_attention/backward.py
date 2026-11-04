def attention_backward(q, k, v, upstream, allowed_mask=None):
    """Return dQ, dK, dV given upstream = dL/dOutput.

    Q, K, and V are independent inputs here. If they share an earlier input,
    its gradient is the sum of the contributions from all three paths.
    """
    q, k, v = _validate_qkv(q, k, v)
    output, weights = scaled_dot_product_attention(q, k, v, allowed_mask)
    upstream = _as_float64(upstream, "upstream")
    if upstream.shape != output.shape:
        raise ValueError(f"upstream must have output shape {output.shape}")

    d_v = weights.swapaxes(-2, -1) @ upstream
    d_weights = upstream @ v.swapaxes(-2, -1)
    # Row-wise softmax Jacobian-vector product, without forming a Jacobian.
    row_dot = np.sum(d_weights * weights, axis=-1, keepdims=True)
    d_scores = weights * (d_weights - row_dot)
    # Masked weights, and therefore their score gradients, are exactly zero.
    scale = np.sqrt(q.shape[-1])
    d_q = (d_scores @ k) / scale
    d_k = (d_scores.swapaxes(-2, -1) @ q) / scale
    return d_q, d_k, d_v
