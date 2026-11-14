def rolling_windows(values, lookback, horizon):
    """Return first-unknown indices, past rows, and future rows."""
    values = _real_array(values, "values", 1)
    lookback, horizon = map(index, (lookback, horizon))
    if min(lookback, horizon) < 1:
        raise ValueError("lookback and horizon must be positive")

    origins = np.arange(lookback, len(values) - horizon + 1)
    if origins.size == 0:
        return origins, np.empty((0, lookback)), np.empty((0, horizon))
    past_indices = origins[:, None] + np.arange(-lookback, 0)
    future_indices = origins[:, None] + np.arange(horizon)
    return origins, values[past_indices], values[future_indices]
