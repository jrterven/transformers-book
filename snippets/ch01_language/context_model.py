def context_logits(context_ids, parameters):
    """Map [..., 2] token IDs to [..., 4] hidden vectors and [..., 16] logits."""
    x = embedding_lookup(parameters["E"], context_ids)
    if x.ndim < 2 or x.shape[-2] != 2:
        raise ValueError("each context must contain exactly two token IDs")
    joined = x.reshape(x.shape[:-2] + (2 * x.shape[-1],))
    hidden = np.maximum(0.0, joined @ parameters["W_context"]
                        + parameters["b_context"])
    logits = hidden @ parameters["W_out"] + parameters["b_out"]
    return hidden, logits
