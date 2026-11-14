def match_queries(cost):
    """Enumerate assignments for a small cost matrix [queries, targets]."""
    cost = _real_array(cost, "cost")
    if cost.ndim != 2:
        raise ValueError("cost needs shape [queries, targets]")
    queries, targets = cost.shape
    if targets > queries:
        raise ValueError("one-to-one assignment needs at least as many queries as targets")
    if math.perm(queries, targets) > 100_000:
        raise ValueError("the teaching search allows at most 100,000 assignments")

    def total(chosen):
        return sum(cost[q, t].item() for t, q in enumerate(chosen))

    # Each candidate lists one distinct query for each target in order.
    chosen = min(permutations(range(queries), targets), key=total)
    pairs = tuple((q, t) for t, q in enumerate(chosen))
    unmatched = tuple(q for q in range(queries) if q not in chosen)
    return pairs, unmatched, total(chosen)
