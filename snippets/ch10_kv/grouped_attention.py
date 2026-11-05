def grouped_attention(q, k, v, allowed=None):
    """Read compact keys and values; masks broadcast to [B, Hq, Q, T]."""
    q, k, v = map(np.asarray, (q, k, v))
    if any(a.ndim != 4 or 0 in a.shape for a in (q, k, v)):
        raise ValueError("q, k, and v need nonempty [batch, head, time, width]")
    if any(a.dtype.kind not in "iuf" for a in (q, k, v)):
        raise ValueError("projected arrays must contain real numbers")
    if any(not np.isfinite(a).all() for a in (q, k, v)):
        raise ValueError("projected arrays must contain only finite values")
    batch, query_heads, query_length, width = q.shape
    kv_heads, key_length = k.shape[1:3]
    if k.shape[:3] != v.shape[:3] or (batch, width) != (k.shape[0], k.shape[-1]):
        raise ValueError("batch, key and value axes, or query/key widths disagree")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")

    group_size = query_heads // kv_heads
    queries = q.astype(np.float64, copy=False).reshape(
        batch, kv_heads, group_size, query_length, width
    )
    keys = k[:, :, None].swapaxes(-1, -2)
    scores = (queries @ keys) / math.sqrt(width)
    flat_scores = scores.reshape(batch, query_heads, query_length, key_length)
    weights = stable_softmax(flat_scores, allowed)
    outputs = weights.reshape(scores.shape) @ v[:, :, None]
    return outputs.reshape(batch, query_heads, query_length, v.shape[-1]), weights
