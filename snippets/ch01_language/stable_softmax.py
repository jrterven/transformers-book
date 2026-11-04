def stable_softmax(logits):
    """Normalize each vocabulary row independently along its last axis."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim < 1 or logits.shape[-1] == 0 or not np.isfinite(logits).all():
        raise ValueError("logits must have a nonempty, finite vocabulary axis")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)
