def causal_decay(a):
    """Build L[t, s] as the product of retentions after write s through t."""
    a = _retentions(a, 1)
    decay = np.eye(len(a))
    for t in range(1, len(a)):
        decay[t, :t] = a[t] * decay[t - 1, :t]
    return decay


def ssd_dense(u, a, b, c, x0=None):
    """Return outputs and the full matrix mapping inputs to outputs."""
    u, b, c, state = _sequence_inputs(u, b, c, x0)
    a = _ssd_retentions(a, len(u))
    with np.errstate(over="raise", invalid="raise"):
        matrix = causal_decay(a) * (c @ b.T)
        from_x0 = np.cumprod(a) * (c @ state)
        outputs = matrix @ u + from_x0
    return outputs, matrix
