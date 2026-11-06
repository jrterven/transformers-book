def fit_seasonal_drift(training, period):
    """Estimate the average change from one season to the next."""
    training = _real_array(training, "training", 1)
    period = index(period)
    if not 1 <= period < len(training):
        raise ValueError("training must span at least one lagged pair")
    with np.errstate(over="raise", invalid="raise"):
        return float((training[period:] - training[:-period]).mean())


def seasonal_drift(history, period, horizon, drift):
    """Add the fitted drift once per forward seasonal cycle."""
    base = seasonal_naive(history, period, horizon)
    drift = _real_array(drift, "drift", 0).item()
    cycles_ahead = 1 + np.arange(len(base)) // index(period)
    with np.errstate(over="raise", invalid="raise"):
        return base + cycles_ahead * drift
