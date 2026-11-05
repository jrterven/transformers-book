def numerical_example():
    annotations = np.array([[1., 0.], [0., 1.], [1., 1.]])
    given_scores = np.log([0.05, 0.10, 0.85])
    weights, context = weighted_read(given_scores, annotations)
    query = np.array([0.2, -0.1])
    scores = additive_scores(query, annotations, np.eye(2), np.eye(2),
                             np.array([1., -1.]))
    scorer_weights, scorer_context = weighted_read(scores, annotations)
    return weights, context, scores, scorer_weights, scorer_context
