def stable_softmax(scores, allowed_mask=None):
    """Normalize the last axis; True mask entries identify allowed keys.

    The mask must broadcast to scores without adding axes. Every row must
    contain an allowed key. Inputs are converted to float64 and must be finite.
    """
    scores = _as_float64(scores, "scores")
    if scores.ndim < 1 or scores.shape[-1] == 0:
        raise ValueError("scores must have a nonempty last axis")
    allowed = _allowed_mask_for(allowed_mask, scores.shape)
    if not np.all(np.any(allowed, axis=-1)):
        raise ValueError("each query must allow at least one key")

    masked_scores = np.where(allowed, scores, -np.inf)
    row_max = np.max(masked_scores, axis=-1, keepdims=True)
    # Subtract before exponentiating. A very negative difference may become
    # -inf in float64; its exponential is correctly zero.
    with np.errstate(over="ignore", under="ignore"):
        numerator = np.exp(masked_scores - row_max)
    return numerator / np.sum(numerator, axis=-1, keepdims=True)
