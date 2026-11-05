def ssd_chunked(u, a, b, c, chunk_size, x0=None):
    """Return outputs and the last state, building only local decay blocks."""
    u, b, c, state = _sequence_inputs(u, b, c, x0)
    a = _ssd_retentions(a, len(u))
    chunk_size = index(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")

    outputs = np.empty(len(u))
    with np.errstate(over="raise", invalid="raise"):
        for start in range(0, len(u), chunk_size):
            rows = slice(start, start + chunk_size)
            u_block, b_block, c_block = u[rows], b[rows], c[rows]
            decay = causal_decay(a[rows])
            prefix = np.cumprod(a[rows])
            within = (decay * (c_block @ b_block.T)) @ u_block
            outputs[rows] = within + prefix * (c_block @ state)

            # The final row gathers all writes ending at this chunk boundary.
            last_write = (decay[-1] * u_block) @ b_block
            state = prefix[-1] * state + last_write
    return outputs, state
