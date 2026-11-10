def delta_sequence(query, key, value, alpha, beta, state=None):
    """Evaluate a sequence; return its outputs and final [value, key] state."""
    query, key, value, alpha, beta, state = _delta_sequence_inputs(
        query, key, value, alpha, beta, state
    )
    outputs = np.empty_like(value)
    for t in range(len(key)):
        outputs[t], state = delta_step(
            state, query[t], key[t], value[t], alpha[t], beta[t]
        )
    return outputs, state
