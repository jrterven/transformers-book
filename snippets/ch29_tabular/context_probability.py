def context_probability(logits, labels):
    """Turn supplied similarities and binary context labels into a toy score."""
    labels = _binary_labels(labels)
    logits = np.asarray(logits)
    if logits.shape != labels.shape or logits.dtype.kind not in "iuf":
        raise ValueError("provide one real similarity per context label")
    weights = stable_softmax(logits)
    # Clipping here absorbs roundoff at the probability endpoints.
    probability = float(np.clip(weights @ labels, 0, 1))
    return probability, weights
