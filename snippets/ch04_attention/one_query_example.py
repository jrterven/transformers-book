def one_query_example(query_index=0):
    """Follow one token's query before doing every query together."""
    query = Q[query_index]
    scores = np.empty(len(K), dtype=np.float64)
    for j in range(len(K)):
        scores[j] = np.sum(query * K[j]) / np.sqrt(len(query))

    weights = stable_softmax(scores)
    contributions = np.empty_like(V)
    output = np.zeros(V.shape[1], dtype=np.float64)
    for j in range(len(V)):
        contributions[j] = weights[j] * V[j]
        output += contributions[j]
    return scores, weights, contributions, output
