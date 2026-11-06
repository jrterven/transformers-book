def delta_step(state, query, key, value, alpha, beta):
    """Apply the supplied query and key vectors and return output and new state."""
    state, query, key, value, alpha, beta = _delta_inputs(
        state, query, key, value, alpha, beta
    )
    with np.errstate(over="raise", invalid="raise"):
        decayed = alpha * state
        prediction = decayed @ key
        error = value - prediction
        correction = beta * np.outer(error, key)
        updated = decayed + correction
        output = updated @ query
    return output, updated
