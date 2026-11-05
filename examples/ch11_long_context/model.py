"""Small NumPy examples of selecting and summarizing attention history."""

from operator import index

import numpy as np

from examples.ch04_attention.attention import (
    scaled_dot_product_attention,
    stable_softmax,
)


# snippet:local_mask
def local_mask(length, radius, global_positions=(), causal=False):
    """Return a [query, key] mask for local and optional global connections."""
    length, radius = index(length), index(radius)
    if length < 1 or radius < 0:
        raise ValueError("length must be positive and radius nonnegative")

    positions = np.arange(length)
    queries, keys = positions[:, None], positions[None, :]
    allowed = np.abs(queries - keys) <= radius
    is_global = np.zeros(length, dtype=bool)
    for position in global_positions:
        position = index(position)
        if not 0 <= position < length:
            raise ValueError("global position is outside the sequence")
        is_global[position] = True
    allowed |= is_global[:, None] | is_global[None, :]
    if causal:
        allowed &= keys <= queries
    return allowed
# end-snippet:local_mask


# snippet:selected_attention
def selected_attention(query, keys, values, index_scores, k, allowed=None):
    """Return one output, the selected positions, and their main weights."""
    query, keys, values, index_scores = map(
        np.asarray, (query, keys, values, index_scores)
    )
    if any(a.dtype.kind not in "iuf" for a in (query, keys, values, index_scores)):
        raise ValueError("queries, keys, values, and index scores must be real numbers")
    if query.ndim != 1 or keys.ndim != 2 or values.ndim != 2:
        raise ValueError("need one query and matrices of key and value rows")
    if keys.shape != (len(values), len(query)):
        raise ValueError("query width or key and value row counts disagree")
    if index_scores.shape != (len(keys),):
        raise ValueError("need one index score per key")
    k = index(k)
    if k < 1:
        raise ValueError("k must be positive")

    if allowed is None:
        allowed = np.ones(len(keys), dtype=bool)
    allowed = np.asarray(allowed)
    if allowed.shape != index_scores.shape or allowed.dtype != np.bool_:
        raise ValueError("allowed must be a boolean vector over key positions")
    candidates = np.flatnonzero(allowed)
    if len(candidates) == 0 or not np.isfinite(index_scores[candidates]).all():
        raise ValueError("need at least one candidate and finite candidate scores")

    # Stable sorting preserves earlier positions when index scores are tied.
    ranked = sorted(candidates, key=lambda position: index_scores[position],
                    reverse=True)
    selected = np.asarray(ranked[:k], dtype=np.intp)
    output, weights = scaled_dot_product_attention(
        query[None, :], keys[selected], values[selected]
    )
    return output[0], selected, weights[0]
# end-snippet:selected_attention


# snippet:block_summaries
def block_summaries(features, gate_logits, stride, overlap=False):
    """Return weighted summaries and the last position in each window."""
    features, gate_logits = map(np.asarray, (features, gate_logits))
    if features.ndim != 2 or features.shape != gate_logits.shape:
        raise ValueError("features and gate logits must share shape [time, width]")
    if features.shape[1] == 0:
        raise ValueError("feature width must be positive")
    if features.dtype.kind not in "iuf" or gate_logits.dtype.kind not in "iuf":
        raise ValueError("features and gate logits must be real numbers")
    features = features.astype(np.float64, copy=False)
    gate_logits = gate_logits.astype(np.float64, copy=False)
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
# end-snippet:block_summaries
