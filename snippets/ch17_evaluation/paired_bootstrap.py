def paired_accuracy_bootstrap(y_true, pred_a, pred_b, *, repeats=20_000,
                              confidence=0.95, seed=17, groups=None):
    """Return B minus A accuracy and a paired percentile interval."""
    truth, a, b = _aligned_predictions(y_true, pred_a, pred_b)
    repeats = index(repeats)
    if repeats < 1 or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("repeats must be positive and confidence between zero and one")

    differences = (b == truth).astype(int) - (a == truth).astype(int)
    totals = differences
    sizes = np.ones(len(truth), dtype=int)
    if groups is not None:
        groups = _group_ids(groups)
        if groups.shape != truth.shape:
            raise ValueError("each row needs one resampling group")
        _, unit = np.unique(groups, return_inverse=True)
        totals = np.bincount(unit, weights=differences)
        sizes = np.bincount(unit)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(totals), size=(repeats, len(totals)))
    replicate_gaps = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(replicate_gaps, [tail, 1 - tail], method="linear")
    return float(differences.mean()), (float(low), float(high))
