def causal_pair_counts(length, window):
    """Count full and windowed causal pairs, including each self-pair."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    _check_window(window)
    width = min(length, window)
    full = length * (length + 1) // 2
    local = length * width - width * (width - 1) // 2
    return full, local
