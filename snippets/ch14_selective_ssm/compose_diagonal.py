def compose_diagonal(first, second):
    """Compose diagonal affine maps by applying first and then second."""
    a1, r1 = first
    a2, r2 = second
    a1, r1, a2, r2 = (
        _real_array(x, name)
        for x, name in ((a1, "a1"), (r1, "r1"), (a2, "a2"), (r2, "r2"))
    )
    if not a1.shape == r1.shape == a2.shape == r2.shape:
        raise ValueError("all diagonal and write arrays must share a shape")
    with np.errstate(over="raise", invalid="raise"):
        return a2 * a1, a2 * r1 + r2
