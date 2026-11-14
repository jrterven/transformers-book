def brier_score(labels, probabilities):
    """Average the squared error of positive-class probabilities."""
    labels = _binary_labels(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != labels.shape or probabilities.dtype.kind not in "iuf":
        raise ValueError("provide one real probability per label")
    if np.any(~np.isfinite(probabilities) | (probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and between zero and one")
    return float(np.mean((probabilities.astype(np.float64, copy=False) - labels) ** 2))
