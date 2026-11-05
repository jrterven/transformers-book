def expert_branch(x, selected, gates, accepted, w_up, w_down):
    """Compute accepted updates and place them back in token order."""
    x = np.asarray(x, dtype=np.float64)
    w_up = np.asarray(w_up, dtype=np.float64)
    w_down = np.asarray(w_down, dtype=np.float64)
    if x.ndim != 2 or w_up.ndim != 3:
        raise ValueError("x needs [tokens, width] and w_up [experts, width, hidden]")
    experts, width, hidden = w_up.shape
    if x.shape[1] != width or w_down.shape != (experts, hidden, width):
        raise ValueError("input and expert widths must agree")
    if any(array.shape != (len(x),) for array in (selected, gates, accepted)):
        raise ValueError("routing arrays need one entry per token")
    if np.any((selected < 0) | (selected >= experts)):
        raise ValueError("each selected ID must index an expert")

    branch = np.zeros_like(x)
    for expert in range(experts):
        rows = np.flatnonzero((selected == expert) & accepted)
        if rows.size == 0:
            continue
        hidden_rows = np.maximum(x[rows] @ w_up[expert], 0)
        branch[rows] = gates[rows, None] * (hidden_rows @ w_down[expert])
    return branch
