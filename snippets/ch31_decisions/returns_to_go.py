def returns_to_go(rewards):
    """Include the current reward in each undiscounted suffix sum."""
    rewards = np.asarray(rewards)
    if rewards.ndim != 1 or rewards.dtype.kind not in "iuf":
        raise ValueError("rewards must be a one-dimensional real sequence")
    with np.errstate(over="ignore", invalid="ignore"):
        targets = np.cumsum(rewards[::-1], dtype=np.float64)[::-1]
    if not np.all(np.isfinite(targets)):
        raise ValueError("rewards and returns must be finite")
    return targets
