def seasonal_naive(history, period, horizon):
    """Repeat the latest observed period into the forecast horizon."""
    history = _real_array(history, "history", 1)
    period, horizon = map(index, (period, horizon))
    if not 1 <= period <= len(history) or horizon < 1:
        raise ValueError("need one observed period and a positive horizon")
    last_season = history[-period:]
    return last_season[np.arange(horizon) % period]
