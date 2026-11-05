def additive_scores(query, annotations, weight_query, weight_annotation,
                    direction, bias=0.0):
    """Return scores [..., S] for annotations [..., S, H].

    The shared direction has shape [A] or [A, 1]; both give one scalar
    per source position. Leading query/annotation batch axes may broadcast.
    """
    direction = np.asarray(direction)
    if direction.ndim == 2 and direction.shape[-1] == 1:
        direction = direction[:, 0]
    if direction.ndim != 1:
        raise ValueError("direction must have shape [A] or [A, 1]")
    keys = np.asarray(annotations) @ weight_annotation
    request = np.asarray(query) @ weight_query
    features = np.tanh(keys + np.expand_dims(request, -2) + bias)
    if direction.shape[0] != features.shape[-1]:
        raise ValueError("direction width must match the score-feature width")
    return features @ direction
