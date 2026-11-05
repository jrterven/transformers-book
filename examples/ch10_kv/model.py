"""Causal decoding on projected query, key, and value arrays."""

import math
from operator import index

import numpy as np

from examples.ch04_attention.attention import stable_softmax


# snippet:causal_mask
def causal_mask(query_start, query_length, key_length):
    """True if a query can read the key, with positions starting at zero."""
    query_start, query_length, key_length = map(
        index, (query_start, query_length, key_length)
    )
    if not 0 <= query_start < query_start + query_length <= key_length:
        raise ValueError("queries must fit inside the key prefix")
    query_positions = query_start + np.arange(query_length)
    key_positions = np.arange(key_length)
    return key_positions[None, :] <= query_positions[:, None]
# end-snippet:causal_mask


# snippet:grouped_attention
def grouped_attention(q, k, v, allowed=None):
    """Return per-head outputs and weights from compact keys and values."""
    q, k, v = map(np.asarray, (q, k, v))
    if any(a.ndim != 4 or 0 in a.shape for a in (q, k, v)):
        raise ValueError("q, k, and v need nonempty [batch, head, time, width]")
    if any(a.dtype.kind not in "iuf" for a in (q, k, v)):
        raise ValueError("projected arrays must contain real numbers")
    batch, query_heads, query_length, width = q.shape
    kv_heads, key_length = k.shape[1:3]
    if k.shape[:3] != v.shape[:3] or (batch, width) != (k.shape[0], k.shape[-1]):
        raise ValueError("batch, key and value axes, or query/key widths disagree")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")

    group_size = query_heads // kv_heads
    queries = q.astype(np.float64, copy=False).reshape(
        batch, kv_heads, group_size, query_length, width
    )
    keys = k[:, :, None].swapaxes(-1, -2)
    scores = (queries @ keys) / math.sqrt(width)
    weights = stable_softmax(scores, allowed)
    outputs = weights @ v[:, :, None]
    return (outputs.reshape(batch, query_heads, query_length, v.shape[-1]),
            weights.reshape(batch, query_heads, query_length, key_length))
# end-snippet:grouped_attention


# snippet:attention_step
def attention_step(q, k_new, v_new, cache=None):
    """Append projected rows and return outputs, weights, and the new cache."""
    q, k_new, v_new = map(np.asarray, (q, k_new, v_new))
    if any(a.ndim != 4 for a in (q, k_new, v_new)):
        raise ValueError("q, k_new, and v_new need four axes")
    query_length = q.shape[-2]
    if not query_length == k_new.shape[-2] == v_new.shape[-2] > 0:
        raise ValueError("each query needs one new key row and value row")

    past_length = 0
    if cache is None:
        keys, values = k_new.copy(), v_new.copy()
    else:
        old_keys, old_values = cache
        past_length = old_keys.shape[-2]
        if old_values.shape[-2] != past_length:
            raise ValueError("cached key and value histories must agree")
        if old_keys.dtype != k_new.dtype or old_values.dtype != v_new.dtype:
            raise ValueError("cache dtype must stay fixed")
        keys = np.concatenate((old_keys, k_new), axis=-2)
        values = np.concatenate((old_values, v_new), axis=-2)

    allowed = causal_mask(past_length, query_length, keys.shape[-2])
    output, weights = grouped_attention(q, keys, values, allowed)
    return output, weights, (keys, values)
# end-snippet:attention_step


# snippet:cache_bytes
def cache_bytes(batch, layers, length, kv_heads, key_width, value_width,
                bytes_per_scalar):
    """Count the K/V array payload, assuming the same dtype in every layer."""
    batch, layers, length, kv_heads, key_width, value_width, bytes_per_scalar = map(
        index, (batch, layers, length, kv_heads, key_width, value_width,
                bytes_per_scalar)
    )
    if length < 0 or min(batch, layers, kv_heads, key_width, value_width,
                         bytes_per_scalar) < 1:
        raise ValueError("cache dimensions must be positive, except empty history")
    per_position = kv_heads * (key_width + value_width) * bytes_per_scalar
    return batch * layers * length * per_position
# end-snippet:cache_bytes
