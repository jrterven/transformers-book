def block_summaries(features, gate_logits, stride, overlap=False):
    """Return weighted summaries and the last position in each window."""
    features = np.asarray(features, dtype=np.float64)
    gate_logits = np.asarray(gate_logits, dtype=np.float64)
    if features.ndim != 2 or features.shape != gate_logits.shape:
        raise ValueError("features and gate logits must share shape [time, width]")
    if features.shape[1] == 0:
        raise ValueError("feature width must be positive")
    if not (np.isfinite(features).all() and np.isfinite(gate_logits).all()):
        raise ValueError("features and gate logits must be finite")
    stride = index(stride)
    if stride < 1:
        raise ValueError("stride must be positive")

    width_in_tokens = 2 * stride if overlap else stride
    ends = np.arange(stride - 1, len(features), stride)
    summaries = np.empty((len(ends), features.shape[1]))
    for row, end in enumerate(ends):
        stop = end + 1
        start = max(0, stop - width_in_tokens)
        window = slice(start, stop)
        # Softmax normalizes the last axis, so transpose to [width, time].
        weights = stable_softmax(gate_logits[window].T).T
        summaries[row] = (weights * features[window]).sum(axis=0)
    return summaries, ends
