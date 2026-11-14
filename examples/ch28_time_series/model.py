"""NumPy teaching examples for chronological forecasting."""

from operator import index

import numpy as np


def _real_array(values, name, ndim):
    values = np.asarray(values)
    if values.ndim != ndim or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real {ndim}-dimensional array")
    with np.errstate(over="raise", invalid="raise"):
        values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


# snippet:rolling_windows
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
# end-snippet:rolling_windows


# snippet:seasonal_naive
def seasonal_naive(history, period, horizon):
    """Repeat the latest observed period into the forecast horizon."""
    history = _real_array(history, "history", 1)
    period, horizon = map(index, (period, horizon))
    if not 1 <= period <= len(history) or horizon < 1:
        raise ValueError("need one observed period and a positive horizon")
    last_season = history[-period:]
    return last_season[np.arange(horizon) % period]
# end-snippet:seasonal_naive


# snippet:seasonal_drift
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
# end-snippet:seasonal_drift


# snippet:forecast_metrics
def forecast_metrics(actual, predicted, lower, upper):
    """Return MAE, inclusive coverage, and average interval width by horizon."""
    actual = _real_array(actual, "actual", 2)
    predicted = _real_array(predicted, "predicted", 2)
    lower = _real_array(lower, "lower", 2)
    upper = _real_array(upper, "upper", 2)
    if actual.size == 0 or not (actual.shape == predicted.shape == lower.shape == upper.shape):
        raise ValueError("use the same nonempty [origins, horizon] shape for all inputs")
    if np.any(lower > upper):
        raise ValueError("lower endpoints must be at most upper endpoints")

    covered = (lower <= actual) & (actual <= upper)
    with np.errstate(over="raise", invalid="raise"):
        mae = np.abs(actual - predicted).mean(axis=0)
        width = (upper - lower).mean(axis=0)
    return mae, covered.mean(axis=0), width
# end-snippet:forecast_metrics
