def ssd_head_chunked(u, a, b, c, chunk_size, x0=None):
    """Evaluate a width-P SSD head; carry one [N, P] boundary state."""
    u, a, b, c, state = _head_inputs(u, a, b, c, x0)
    chunk_size = index(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    outputs = np.empty_like(u)
    with np.errstate(over="raise", invalid="raise"):
        for start in range(0, len(u), chunk_size):
            rows = slice(start, start + chunk_size)
            ub, bb, cb = u[rows], b[rows], c[rows]
            decay = causal_decay(a[rows])
            prefix = np.cumprod(a[rows])
            within = (decay * (cb @ bb.T)) @ ub
            outputs[rows] = within + prefix[:, None] * (cb @ state)
            last_write = bb.T @ (decay[-1, :, None] * ub)
            state = prefix[-1] * state + last_write
    return outputs, state
