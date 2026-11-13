"""NumPy teaching examples for tables and in-context weighting."""

from numbers import Real

import numpy as np

from examples.ch04_attention.attention import stable_softmax


def _numeric_column(values):
    if isinstance(values, (str, bytes)):
        raise TypeError("provide a sequence of numeric values")
    numbers = []
    for value in values:
        if value is None:
            numbers.append(np.nan)
        elif isinstance(value, Real) and not isinstance(value, bool):
            numbers.append(float(value))
        else:
            raise ValueError("numeric values must be real numbers or missing")
    numbers = np.array(numbers, dtype=np.float64)
    if np.isinf(numbers).any():
        raise ValueError("numeric values must be finite or missing")
    return numbers


def _category_column(values):
    if isinstance(values, (str, bytes)):
        raise TypeError("provide a sequence of categorical values")
    categories = tuple(values)
    if any(value is not None and not isinstance(value, str) for value in categories):
        raise ValueError("categories must be strings or None")
    return categories


def _columns(numeric, categories):
    numeric = _numeric_column(numeric)
    categories = _category_column(categories)
    if len(numeric) != len(categories):
        raise ValueError("numeric and categorical columns need the same row count")
    return numeric, categories


# snippet:fit_row_encoder
def fit_row_encoder(numeric, categories):
    """Fit a numeric median and a sorted categorical vocabulary on training rows."""
    numeric, categories = _columns(numeric, categories)
    observed = numeric[~np.isnan(numeric)]
    if observed.size == 0:
        raise ValueError("fit needs at least one observed numeric training value")
    with np.errstate(over="raise", invalid="raise"):
        median = float(np.median(observed))
    levels = tuple(sorted({value for value in categories if value is not None}))
    return median, levels
# end-snippet:fit_row_encoder


# snippet:encode_rows
def encode_rows(numeric, categories, median, levels):
    """Apply fitted values and keep separate missing and unknown indicators."""
    numeric, categories = _columns(numeric, categories)
    median = _numeric_column((median,))[0]
    if np.isnan(median):
        raise ValueError("median must be finite")
    levels = _category_column(levels)
    if None in levels or len(set(levels)) != len(levels):
        raise ValueError("levels must be unique observed category strings")

    features = np.zeros((len(numeric), 4 + len(levels)))
    numeric_missing = np.isnan(numeric)
    features[:, 0] = np.where(numeric_missing, median, numeric)
    features[:, 1] = numeric_missing
    level_column = {value: 2 + i for i, value in enumerate(levels)}
    for row, value in enumerate(categories):
        if value is None:
            features[row, -2] = 1
        elif value in level_column:
            features[row, level_column[value]] = 1
        else:
            features[row, -1] = 1
    return features
# end-snippet:encode_rows


def _binary_labels(labels):
    labels = np.asarray(labels)
    if (labels.ndim != 1 or labels.size == 0 or labels.dtype.kind not in "biuf"
            or np.any((labels != 0) & (labels != 1))):
        raise ValueError("labels must be a nonempty binary vector")
    return labels.astype(np.float64, copy=False)


# snippet:context_probability
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
# end-snippet:context_probability


# snippet:brier_score
def brier_score(labels, probabilities):
    """Average the squared error of positive-class probabilities."""
    labels = _binary_labels(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != labels.shape or probabilities.dtype.kind not in "iuf":
        raise ValueError("provide one real probability per label")
    if np.any(~np.isfinite(probabilities) | (probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and between zero and one")
    return float(np.mean((probabilities.astype(np.float64, copy=False) - labels) ** 2))
# end-snippet:brier_score


def histogram_summary(edges, mass, levels):
    """Return means and quantiles assuming uniform density inside each bin."""
    edges = np.asarray(edges, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    if (edges.ndim != 1 or mass.ndim != 2 or mass.shape[1] == 0
            or len(edges) != mass.shape[1] + 1 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("provide ordered finite edges for each probability bin")
    if (not np.isfinite(mass).all() or np.any(mass < 0)
            or not np.allclose(mass.sum(axis=1), 1)):
        raise ValueError("each row of bin masses must be a probability distribution")
    if levels.ndim != 1 or not np.all((0 < levels) & (levels < 1)):
        raise ValueError("provide quantile levels strictly between zero and one")

    mass = mass / mass.sum(axis=1, keepdims=True)
    centers = (edges[:-1] + edges[1:]) / 2
    means = mass @ centers
    cumulative = np.column_stack((np.zeros(len(mass)), np.cumsum(mass, axis=1)))
    cumulative /= cumulative[:, -1:]
    quantiles = np.empty((len(mass), len(levels)))
    row = np.arange(len(mass))
    for j, level in enumerate(levels):
        bin_index = np.sum(cumulative[:, 1:] < level, axis=1)
        below = cumulative[row, bin_index]
        above = cumulative[row, bin_index + 1]
        fraction = (level - below) / (above - below)
        left, right = edges[bin_index], edges[bin_index + 1]
        quantiles[:, j] = left + fraction * (right - left)
    return means, quantiles
