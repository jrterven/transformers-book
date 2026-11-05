def zoh_coefficients(delta):
    """Return retain and write coefficients for dx/dt = -x + u."""
    delta = _real_array(delta, "delta")
    if np.any(delta <= 0):
        raise ValueError("steps must be positive")
    with np.errstate(under="ignore"):
        retain = np.exp(-delta)
        write = -np.expm1(-delta)
    return retain, write
