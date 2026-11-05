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
    return output, weights
