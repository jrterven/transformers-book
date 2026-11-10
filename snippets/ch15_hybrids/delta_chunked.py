def delta_chunked(query, key, value, alpha, beta, chunk_size, state=None):
    """Solve local error dependencies and pass only the final matrix state."""
    query, key, value, alpha, beta, state = _delta_sequence_inputs(
        query, key, value, alpha, beta, state
    )
    chunk_size = index(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    outputs = np.empty_like(value)
    with np.errstate(over="raise", invalid="raise"):
        for start in range(0, len(key), chunk_size):
            rows = slice(start, start + chunk_size)
            q, k, v, a, b = (x[rows] for x in (query, key, value, alpha, beta))
            decay = causal_decay(a)
            prefix = np.cumprod(a)
            dependencies = decay * (k @ k.T) * b[None, :]
            system = np.eye(len(k)) + np.tril(dependencies, k=-1)
            residual = v - prefix[:, None] * (k @ state.T)
            errors = np.linalg.solve(system, residual)
            reads = decay * (q @ k.T) * b[None, :]
            outputs[rows] = prefix[:, None] * (q @ state.T) + reads @ errors
            writes = (errors * (decay[-1] * b)[:, None]).T @ k
            state = prefix[-1] * state + writes
    return outputs, state
