def contact_map(coordinates, threshold, min_sequence_separation=1):
    """Return a symmetric Boolean matrix of contacts under two cutoffs."""
    coordinates, threshold, min_sequence_separation = _contact_inputs(
        coordinates, threshold, min_sequence_separation
    )
    # Scale each pair and its threshold together before squaring.
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        difference = coordinates[:, None, :] - coordinates[None, :, :]
        largest = np.maximum(np.max(np.abs(difference), axis=-1), threshold)
        _, exponent = np.frexp(largest)
        scaled = np.ldexp(difference, -exponent[..., None])
        cutoff = np.ldexp(threshold, -exponent)
        within = np.isfinite(largest) & ((scaled**2).sum(axis=-1) <= cutoff**2)
    indices = np.arange(len(coordinates))
    separation = np.abs(indices[:, None] - indices[None, :])
    return within & (separation >= min_sequence_separation)
