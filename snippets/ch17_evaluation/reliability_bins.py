def reliability_bins(y_true, positive_scores, edges):
    """Return bin counts, mean scores, positive rates, and the Brier score."""
    truth = _binary_vector(y_true, "truth")
    scores = _probability_vector(positive_scores, "scores")
    edges = _probability_vector(edges, "edges")
    if scores.shape != truth.shape:
        raise ValueError("each row needs one positive-class score")
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin edges must increase strictly from zero to one")

    n_bins = len(edges) - 1
    # Interior edges begin their next bin. The final bin includes one.
    bin_ids = np.minimum(np.searchsorted(edges, scores, side="right") - 1, n_bins - 1)
    counts = np.bincount(bin_ids, minlength=n_bins)
    mean_scores = np.full(n_bins, np.nan)
    positive_rates = np.full(n_bins, np.nan)
    for bin_index in np.flatnonzero(counts):
        in_bin = bin_ids == bin_index
        mean_scores[bin_index] = scores[in_bin].mean()
        positive_rates[bin_index] = truth[in_bin].mean()
    brier = float(np.mean((scores - truth) ** 2))
    return counts, mean_scores, positive_rates, brier
