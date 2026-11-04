"""Float64 attention with explicit sequence, feature, batch, and head axes."""

import numpy as np


def _as_float64(value, name):
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _allowed_mask_for(allowed_mask, shape):
    if allowed_mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(allowed_mask)
    if mask.dtype != np.bool_:
        raise TypeError("allowed_mask must be boolean: True means allowed")
    try:
        return np.broadcast_to(mask, shape)
    except ValueError as error:
        raise ValueError(
            f"allowed_mask shape {mask.shape} cannot broadcast to scores {shape}"
        ) from error


# snippet:stable_softmax
def stable_softmax(scores, allowed_mask=None):
    """Normalize the last axis; True mask entries identify allowed keys.

    The mask must broadcast to scores without adding axes. Every row must
    contain an allowed key. Inputs are converted to float64 and must be finite.
    """
    scores = _as_float64(scores, "scores")
    if scores.ndim < 1 or scores.shape[-1] == 0:
        raise ValueError("scores must have a nonempty last axis")
    allowed = _allowed_mask_for(allowed_mask, scores.shape)
    if not np.all(np.any(allowed, axis=-1)):
        raise ValueError("each query must allow at least one key")

    masked_scores = np.where(allowed, scores, -np.inf)
    row_max = np.max(masked_scores, axis=-1, keepdims=True)
    # Subtract before exponentiating. A very negative difference may become
    # -inf in float64; its exponential is correctly zero.
    with np.errstate(over="ignore", under="ignore"):
        numerator = np.exp(masked_scores - row_max)
    return numerator / np.sum(numerator, axis=-1, keepdims=True)
# end-snippet:stable_softmax


def _validate_qkv(q, k, v):
    q = _as_float64(q, "q")
    k = _as_float64(k, "k")
    v = _as_float64(v, "v")
    if min(q.ndim, k.ndim, v.ndim) < 2:
        raise ValueError("q, k, and v need sequence and feature axes")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v must have identical leading axes")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same feature dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same number of tokens")
    if any(size == 0 for array in (q, k, v) for size in array.shape):
        raise ValueError("q, k, and v dimensions must be nonempty")
    return q, k, v


# snippet:attention_core
def scaled_dot_product_attention(q, k, v, allowed_mask=None):
    """Return (output, weights) for q (..., Nq, Dk), k (..., Nk, Dk),
    and v (..., Nk, Dv). Leading axes must match exactly.

    Weights have shape (..., Nq, Nk); output has shape (..., Nq, Dv).
    A boolean mask follows NumPy broadcasting; True means allowed.
    """
    q, k, v = _validate_qkv(q, k, v)
    scale = np.sqrt(q.shape[-1])
    scores = (q @ k.swapaxes(-2, -1)) / scale
    weights = stable_softmax(scores, allowed_mask)
    output = weights @ v
    return output, weights
# end-snippet:attention_core


# snippet:backward
def attention_backward(q, k, v, upstream, allowed_mask=None):
    """Return dQ, dK, dV given upstream = dL/dOutput.

    Q, K, and V are independent inputs here. If they share an earlier input,
    its gradient is the sum of the contributions from all three paths.
    """
    q, k, v = _validate_qkv(q, k, v)
    output, weights = scaled_dot_product_attention(q, k, v, allowed_mask)
    upstream = _as_float64(upstream, "upstream")
    if upstream.shape != output.shape:
        raise ValueError(f"upstream must have output shape {output.shape}")

    d_v = weights.swapaxes(-2, -1) @ upstream
    d_weights = upstream @ v.swapaxes(-2, -1)
    # Row-wise softmax Jacobian-vector product, without forming a Jacobian.
    row_dot = np.sum(d_weights * weights, axis=-1, keepdims=True)
    d_scores = weights * (d_weights - row_dot)
    # Masked weights, and therefore their score gradients, are exactly zero.
    scale = np.sqrt(q.shape[-1])
    d_q = (d_scores @ k) / scale
    d_k = (d_scores.swapaxes(-2, -1) @ q) / scale
    return d_q, d_k, d_v
# end-snippet:backward
