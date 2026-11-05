"""A one-head NumPy reference for online softmax and tiled attention."""

import math
from operator import index

import numpy as np


def _real_matrix(array, name):
    array = np.asarray(array)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


# snippet:online_update
def online_update(maximum, denominator, numerator, scores, values, allowed=None):
    """Advance a row state initialized with maximum -inf and zero sums."""
    scores = _real_matrix(scores, "scores")
    values = _real_matrix(values, "values")
    rows, keys = scores.shape
    if (maximum.shape != (rows,) or denominator.shape != (rows,)
            or numerator.shape != (rows, values.shape[1])
            or values.shape[0] != keys):
        raise ValueError("state and value axes must agree with the score tile")
    if allowed is not None:
        allowed = np.asarray(allowed)
        if allowed.shape != scores.shape or allowed.dtype != np.bool_:
            raise ValueError("allowed must be a boolean mask with the score shape")
        scores = np.where(allowed, scores, -np.inf)

    new_maximum = np.maximum(maximum, scores.max(axis=1))
    # Use zero only as a subtraction reference for rows with no allowed key.
    reference = np.where(np.isneginf(new_maximum), 0.0, new_maximum)
    with np.errstate(over="ignore", under="ignore"):
        old_scale = np.exp(maximum - reference)
        weights = np.exp(scores - reference[:, None])
    with np.errstate(over="raise", invalid="raise"):
        denominator = old_scale * denominator + weights.sum(axis=1)
        numerator = old_scale[:, None] * numerator + weights @ values
    return new_maximum, denominator, numerator
# end-snippet:online_update


def _tiled_inputs(q, k, v, query_tile, key_tile, causal, query_start):
    q = _real_matrix(q, "q")
    k = _real_matrix(k, "k")
    v = _real_matrix(v, "v")
    if q.shape[1] != k.shape[1] or len(k) != len(v):
        raise ValueError("query and key widths or key and value lengths disagree")
    query_tile, key_tile, query_start = map(index, (query_tile, key_tile, query_start))
    if min(query_tile, key_tile) < 1:
        raise ValueError("tile sizes must be positive")
    if query_start < 0 or (causal and query_start + len(q) > len(k)):
        raise ValueError("causal queries must fit inside the key prefix")
    return q, k, v, query_tile, key_tile, query_start


# snippet:tiled_attention
def tiled_attention(q, k, v, query_tile, key_tile, *, causal=False, query_start=0):
    """Return one output row per query, using compact score tiles."""
    q, k, v, query_tile, key_tile, query_start = _tiled_inputs(
        q, k, v, query_tile, key_tile, causal, query_start
    )
    output = np.empty((len(q), v.shape[1]))
    scale = math.sqrt(q.shape[1])
    for q_start in range(0, len(q), query_tile):
        queries = q[q_start:q_start + query_tile]
        positions = query_start + q_start + np.arange(len(queries))
        maximum = np.full(len(queries), -np.inf)
        denominator = np.zeros(len(queries))
        numerator = np.zeros((len(queries), v.shape[1]))
        for k_start in range(0, len(k), key_tile):
            keys = k[k_start:k_start + key_tile]
            values = v[k_start:k_start + key_tile]
            allowed = None
            if causal:
                key_positions = k_start + np.arange(len(keys))
                allowed = key_positions[None, :] <= positions[:, None]
                if not allowed.any():
                    continue
            scores = (queries @ keys.T) / scale
            maximum, denominator, numerator = online_update(
                maximum, denominator, numerator, scores, values, allowed
            )
        output[q_start:q_start + len(queries)] = numerator / denominator[:, None]
    return output
# end-snippet:tiled_attention


# snippet:score_workspace
def score_workspace(query_length, key_length, query_tile, key_tile):
    """Return element counts for one dense score array and one largest tile."""
    query_length, key_length, query_tile, key_tile = map(
        index, (query_length, key_length, query_tile, key_tile)
    )
    if min(query_length, key_length, query_tile, key_tile) < 1:
        raise ValueError("sequence lengths and tile sizes must be positive")
    full = query_length * key_length
    largest_tile = min(query_length, query_tile) * min(key_length, key_tile)
    return full, largest_tile
# end-snippet:score_workspace
