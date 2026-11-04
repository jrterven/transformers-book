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
