"""NumPy teaching calculations for a small binary evaluation."""

import math
from operator import index

import numpy as np


def _group_ids(group_ids):
    ids = np.asarray(group_ids)
    if ids.ndim != 1 or len(ids) == 0 or ids.dtype.kind not in "iuUS":
        raise ValueError("group IDs must be a nonempty vector of integers or strings")
    if ids.dtype.kind in "US" and np.any(np.char.str_len(ids) == 0):
        raise ValueError("group IDs must not be empty strings")
    return ids


# snippet:group_split
def split_by_group(group_ids, group_counts, *, seed=17):
    """Return train, validation, and test row indices with disjoint groups."""
    group_ids = _group_ids(group_ids)
    names = np.unique(group_ids)
    train_count, validation_count, test_count = map(index, group_counts)
    counts = (train_count, validation_count, test_count)
    if min(counts) < 1 or sum(counts) != len(names):
        raise ValueError("three positive group counts must sum to the number of groups")

    shuffled = np.random.default_rng(seed).permutation(names)
    parts = np.split(shuffled, [train_count, train_count + validation_count])
    return tuple(np.flatnonzero(np.isin(group_ids, part)) for part in parts)
# end-snippet:group_split


def _binary_vector(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0 or values.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a nonempty binary vector")
    if np.any((values != 0) & (values != 1)):
        raise ValueError(f"{name} entries must be 0 or 1")
    return values.astype(bool, copy=False)


def _aligned_predictions(y_true, *predictions):
    truth = _binary_vector(y_true, "truth")
    predicted = tuple(_binary_vector(p, "predictions") for p in predictions)
    if any(p.shape != truth.shape for p in predicted):
        raise ValueError("truth and predictions must refer to the same rows")
    return truth, *predicted


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


# snippet:binary_metrics
def binary_metrics(y_true, y_pred):
    """Count the four outcomes and compute metrics on aligned rows."""
    truth, pred = _aligned_predictions(y_true, y_pred)
    tp = int(np.count_nonzero(truth & pred))
    fn = int(np.count_nonzero(truth & ~pred))
    fp = int(np.count_nonzero(~truth & pred))
    tn = int(np.count_nonzero(~truth & ~pred))

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": (tp + tn) / len(truth),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }
# end-snippet:binary_metrics


def _probability_vector(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a vector of real numbers")
    values = values.astype(np.float64, copy=False)
    if np.any(~np.isfinite(values) | (values < 0) | (values > 1)):
        raise ValueError(f"{name} must be finite and between zero and one")
    return values


# snippet:reliability_bins
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
# end-snippet:reliability_bins


# snippet:paired_bootstrap
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
# end-snippet:paired_bootstrap
