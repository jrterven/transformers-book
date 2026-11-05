def additive_scores(query, annotations, weight_query, weight_annotation,
                    direction, bias=0.0):
    """NumPy row-vector calculation, with optional leading batch axes."""
    keys = np.asarray(annotations) @ weight_annotation
    request = np.asarray(query) @ weight_query
    features = np.tanh(keys + np.expand_dims(request, -2) + bias)
    return features @ direction
