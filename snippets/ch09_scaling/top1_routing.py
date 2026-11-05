def route_top1(logits, capacity):
    """Return probabilities, zero-based expert IDs, gates, and acceptance."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or min(logits.shape) == 0:
        raise ValueError("logits must have shape [tokens, experts]")
    if (isinstance(capacity, (bool, np.bool_))
            or not isinstance(capacity, (int, np.integer)) or capacity < 0):
        raise ValueError("capacity must be a nonnegative integer")

    probabilities = stable_softmax(logits)
    # The largest logit gives the largest probability before rounding.
    selected = logits.argmax(axis=1)
    gates = probabilities[np.arange(len(selected)), selected]
    accepted = np.zeros(len(selected), dtype=bool)
    for expert in range(probabilities.shape[1]):
        requests = np.flatnonzero(selected == expert)
        accepted[requests[:capacity]] = True
    return probabilities, selected, gates, accepted
