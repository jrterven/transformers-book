def pass_at_k(n, correct, k):
    """Probability a uniform size-k subset includes a passing candidate."""
    if any(isinstance(value, bool) for value in (n, correct, k)):
        raise TypeError("n, correct, and k must be integer counts")
    n, correct, k = map(index, (n, correct, k))
    if not 1 <= k <= n or not 0 <= correct <= n:
        raise ValueError("use 1 <= k <= n and 0 <= correct <= n")
    total = comb(n, k)
    misses = comb(n - correct, k)
    return (total - misses) / total
