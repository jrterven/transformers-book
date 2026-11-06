def sentence_logits(features, valid, weight, bias):
    """Return masked means and logits for one sentence or a batch."""
    features = _real_array(features, "features")
    weight = _real_array(weight, "weight")
    bias = _real_array(bias, "bias")
    valid = np.asarray(valid)
    if features.ndim not in (2, 3) or min(features.shape) == 0:
        raise ValueError("features need shape [time, width] or [batch, time, width]")
    if valid.shape != features.shape[:-1] or valid.dtype != np.bool_:
        raise ValueError("valid must be a boolean mask matching the token axes")
    if (weight.ndim != 2 or min(weight.shape) == 0
            or weight.shape[1] != features.shape[-1] or bias.shape != weight.shape[:1]):
        raise ValueError("weight needs [classes, width] and bias needs [classes]")
    count = valid.sum(axis=-1)
    if np.any(count == 0):
        raise ValueError("each sentence must select at least one token")
    if not all(np.isfinite(x).all() for x in (features[valid], weight, bias)):
        raise ValueError("selected features and head parameters must be finite")

    with np.errstate(over="raise", invalid="raise"):
        kept = np.where(valid[..., None], features, 0)
        pooled = kept.sum(axis=-2) / count[..., None]
        logits = pooled @ weight.T + bias
    return pooled, logits
