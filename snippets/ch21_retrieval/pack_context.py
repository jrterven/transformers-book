def pack_context(ranked_ids, corpus, token_budget):
    """Pack whole passages, preserving IDs and the rank order of kept items."""
    ids = _passage_ids(ranked_ids)
    token_budget = index(token_budget)
    if token_budget < 0:
        raise ValueError("token budget must be nonnegative")
    packed = []
    for pid in ids:
        text = corpus[pid]
        cost = len(tokenize(text))
        if cost <= token_budget:
            packed.append((pid, text))
            token_budget -= cost
    return tuple(packed)
