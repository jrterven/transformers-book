def local_mask(length, radius, global_positions=(), causal=False):
    """Return a [query, key] mask for local and optional global connections."""
    length, radius = index(length), index(radius)
    if length < 1 or radius < 0:
        raise ValueError("length must be positive and radius nonnegative")

    positions = np.arange(length)
    queries, keys = positions[:, None], positions[None, :]
    allowed = np.abs(queries - keys) <= radius
    is_global = np.zeros(length, dtype=bool)
    for position in global_positions:
        position = index(position)
        if not 0 <= position < length:
            raise ValueError("global position is outside the sequence")
        is_global[position] = True
    allowed |= is_global[:, None] | is_global[None, :]
    if causal:
        allowed &= keys <= queries
    return allowed
