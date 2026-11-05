def selective_scan(u, a, b, c, x0=None):
    """Return outputs and states for a precomputed diagonal update."""
    u, b, c, state = _sequence_inputs(u, b, c, x0)
    a = _retentions(a, 2)
    if a.shape not in (b.shape, (len(u), 1)):
        raise ValueError("a must have one or state-width retentions per time step")

    outputs = np.empty(len(u))
    states = np.empty_like(b)
    with np.errstate(over="raise", invalid="raise"):
        for t, token in enumerate(u):
            state = a[t] * state + b[t] * token
            states[t] = state
            outputs[t] = c[t] @ state
    return outputs, states
