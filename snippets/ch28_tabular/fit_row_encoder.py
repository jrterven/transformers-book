def fit_row_encoder(numeric, categories):
    """Fit a numeric median and a sorted categorical vocabulary on training rows."""
    numeric, categories = _columns(numeric, categories)
    observed = numeric[~np.isnan(numeric)]
    if observed.size == 0:
        raise ValueError("fit needs at least one observed numeric training value")
    with np.errstate(over="raise", invalid="raise"):
        median = float(np.median(observed))
    levels = tuple(sorted({value for value in categories if value is not None}))
    return median, levels
