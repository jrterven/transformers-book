def bilinear_discretize(a, b, delta):
    """Compute the discrete transition and input map of a continuous system."""
    a, b = _transition_and_input(a, b)
    delta = _real_array(delta, "delta", 0).item()
    if delta <= 0:
        raise ValueError("delta must be positive")

    identity = np.eye(len(b))
    with np.errstate(over="raise", invalid="raise"):
        half_step = (delta / 2) * a
        left = identity - half_step
        a_bar = np.linalg.solve(left, identity + half_step)
        b_bar = np.linalg.solve(left, delta * b)
    if not (np.isfinite(a_bar).all() and np.isfinite(b_bar).all()):
        raise FloatingPointError("discretization produced a nonfinite result")
    return a_bar, b_bar
