"""Two evaluation orders for one MLA query position in NumPy."""

import numpy as np

from examples.ch04_attention.attention import (
    scaled_dot_product_attention,
    stable_softmax,
)


def _validate_inputs(q_content, q_rotary, latent, k_rotary, w_uk, w_uv):
    arrays = tuple(map(np.asarray, (q_content, q_rotary, latent, k_rotary, w_uk, w_uv)))
    if any(x.ndim != rank or x.size == 0
           for x, rank in zip(arrays, (2, 2, 2, 2, 3, 3))):
        raise ValueError("queries and histories need rank 2; projections need rank 3")
    if any(x.dtype.kind not in "iuf" for x in arrays):
        raise ValueError("queries, histories, and projections must contain real numbers")
    arrays = tuple(x.astype(np.float64, copy=False) for x in arrays)
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("all inputs must be finite")

    q_content, q_rotary, latent, k_rotary, w_uk, w_uv = arrays
    heads, content_width = q_content.shape
    tokens, latent_width = latent.shape
    if q_rotary.shape[0] != heads or k_rotary.shape != (tokens, q_rotary.shape[1]):
        raise ValueError("rotary queries and keys must match the head and history axes")
    if w_uk.shape != (heads, latent_width, content_width):
        raise ValueError("key projections must have shape [heads, latent, content]")
    if w_uv.shape[:2] != (heads, latent_width):
        raise ValueError("value projections must have shape [heads, latent, value]")
    return arrays


def expanded_attention(q_content, q_rotary, latent, k_rotary, w_uk, w_uv):
    """Expand each head's content keys and values for the supplied prefix."""
    q_content, q_rotary, latent, k_rotary, w_uk, w_uv = _validate_inputs(
        q_content, q_rotary, latent, k_rotary, w_uk, w_uv
    )
    k_content = latent @ w_uk
    values = latent @ w_uv
    # This view exposes the same rotary key history to each query head.
    rotary_by_head = np.broadcast_to(k_rotary, (len(q_content), *k_rotary.shape))
    queries = np.concatenate((q_content, q_rotary), axis=-1)
    keys = np.concatenate((k_content, rotary_by_head), axis=-1)
    output, weights = scaled_dot_product_attention(queries[:, None, :], keys, values)
    return output[:, 0, :], weights[:, 0, :]


# snippet:latent_attention
def latent_attention(q_content, q_rotary, latent, k_rotary, w_uk, w_uv):
    """Return [heads, value] outputs and [heads, tokens] weights."""
    q_content, q_rotary, latent, k_rotary, w_uk, w_uv = _validate_inputs(
        q_content, q_rotary, latent, k_rotary, w_uk, w_uv
    )
    # Keep the same scale as the expanded content-plus-rotary dot product.
    scale = np.sqrt(q_content.shape[1] + q_rotary.shape[1])
    q_latent = (q_content[:, None, :] @ w_uk.swapaxes(-1, -2))[:, 0, :]
    content_scores = q_latent @ latent.T
    rotary_scores = q_rotary @ k_rotary.T
    weights = stable_softmax((content_scores + rotary_scores) / scale)

    # Mix the shared latent rows first, then apply each head's value map.
    mixed_latent = weights @ latent
    output = (mixed_latent[:, None, :] @ w_uv)[:, 0, :]
    if not np.isfinite(output).all():
        raise ValueError("attention output must be finite")
    return output, weights
# end-snippet:latent_attention


def latent_inputs():
    """Return fresh arrays for the two-head, two-position teaching example."""
    q_content = np.array([[2., 1.], [1., -1.]])
    q_rotary = np.array([[0., 1.], [1., 0.]])
    latent = np.array([[1., 2.], [2., 0.]])
    k_rotary = np.eye(2)
    w_uk = np.array([[[1., 0.], [1., 1.]], [[0., 1.], [1., 0.]]])
    w_uv = np.array([[[1., 0.], [0., 2.]], [[0., 1.], [2., 0.]]])
    return q_content, q_rotary, latent, k_rotary, w_uk, w_uv


def latent_demo():
    """Compare the two orders and the chapter's separate hand calculations."""
    inputs = latent_inputs()
    q_content, _, latent, k_rotary, w_uk, w_uv = inputs
    expanded_output, expanded_weights = expanded_attention(*inputs)
    latent_output, latent_weights = latent_attention(*inputs)

    heads, content_width = q_content.shape
    value_width = w_uv.shape[-1]
    expanded_content_scalars = len(latent) * heads * (content_width + value_width)
    q = q_content[0]
    hand_weights = np.array([.25, .75])
    return {
        "cached_scalars": latent.size + k_rotary.size,
        "expanded_scalars_with_shared_rotary": expanded_content_scalars + k_rotary.size,
        "expanded_output": expanded_output.tolist(),
        "latent_output": latent_output.tolist(),
        "expanded_weights": expanded_weights.tolist(),
        "latent_weights": latent_weights.tolist(),
        "max_output_difference": float(np.max(np.abs(expanded_output - latent_output))),
        "max_weight_difference": float(np.max(np.abs(expanded_weights - latent_weights))),
        "hand_checks": {
            "expanded_content_dot": float(q @ (latent[0] @ w_uk[0])),
            "absorbed_content_dot": float((q @ w_uk[0].T) @ latent[0]),
            "expanded_weighted_value": (hand_weights @ (latent @ w_uv[0])).tolist(),
            "absorbed_weighted_value": ((hand_weights @ latent) @ w_uv[0]).tolist(),
        },
    }
