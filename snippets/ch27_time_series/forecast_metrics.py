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
