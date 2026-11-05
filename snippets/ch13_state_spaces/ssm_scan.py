def ssm_scan(u, a_bar, b_bar, c, x0=None):
    """Return outputs [time] and the states after each update [time, width]."""
    u = _real_array(u, "u", 1)
    a_bar, b_bar, c = _state_space(a_bar, b_bar, c)
    state = np.zeros_like(b_bar) if x0 is None else _real_array(x0, "x0", 1)
    if state.shape != b_bar.shape:
        raise ValueError("x0 must match the state width")

    outputs = np.empty(len(u))
    states = np.empty((len(u), len(state)))
    with np.errstate(over="raise", invalid="raise"):
        for t, token in enumerate(u):
            state = a_bar @ state + b_bar * token
            states[t] = state
            outputs[t] = c @ state
    return outputs, states
