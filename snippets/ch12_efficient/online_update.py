def online_update(maximum, denominator, numerator, scores, values, allowed=None):
    """Advance a row state initialized with maximum -inf and zero sums."""
    scores = _real_matrix(scores, "scores")
    values = _real_matrix(values, "values")
    rows, keys = scores.shape
    if (maximum.shape != (rows,) or denominator.shape != (rows,)
            or numerator.shape != (rows, values.shape[1])
            or values.shape[0] != keys):
        raise ValueError("state and value axes must agree with the score tile")
    if allowed is not None:
        allowed = np.asarray(allowed)
        if allowed.shape != scores.shape or allowed.dtype != np.bool_:
            raise ValueError("allowed must be a boolean mask with the score shape")
        scores = np.where(allowed, scores, -np.inf)

    new_maximum = np.maximum(maximum, scores.max(axis=1))
    # Use zero only as a subtraction reference for rows with no allowed key.
    reference = np.where(np.isneginf(new_maximum), 0.0, new_maximum)
    with np.errstate(over="ignore", under="ignore"):
        old_scale = np.exp(maximum - reference)
        weights = np.exp(scores - reference[:, None])
    with np.errstate(over="raise", invalid="raise"):
        denominator = old_scale * denominator + weights.sum(axis=1)
        numerator = old_scale[:, None] * numerator + weights @ values
    return new_maximum, denominator, numerator
