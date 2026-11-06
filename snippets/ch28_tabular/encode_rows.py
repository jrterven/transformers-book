def encode_rows(numeric, categories, median, levels):
    """Apply fitted values and keep separate missing and unknown indicators."""
    numeric, categories = _columns(numeric, categories)
    median = _numeric_column((median,))[0]
    if np.isnan(median):
        raise ValueError("median must be finite")
    levels = _category_column(levels)
    if None in levels or len(set(levels)) != len(levels):
        raise ValueError("levels must be unique observed category strings")

    features = np.zeros((len(numeric), 4 + len(levels)))
    numeric_missing = np.isnan(numeric)
    features[:, 0] = np.where(numeric_missing, median, numeric)
    features[:, 1] = numeric_missing
    level_column = {value: 2 + i for i, value in enumerate(levels)}
    for row, value in enumerate(categories):
        if value is None:
            features[row, -2] = 1
        elif value in level_column:
            features[row, level_column[value]] = 1
        else:
            features[row, -1] = 1
    return features
