def ssm_kernel(a_bar, b_bar, c, length):
    """Read the response to a unit impulse at lags zero through length - 1."""
    a_bar, b_bar, c = _state_space(a_bar, b_bar, c)
    length = index(length)
    if length < 1:
        raise ValueError("kernel length must be positive")

    impulse = b_bar
    kernel = np.empty(length)
    with np.errstate(over="raise", invalid="raise"):
        kernel[0] = c @ impulse
        for lag in range(1, length):
            impulse = a_bar @ impulse
            kernel[lag] = c @ impulse
    return kernel
