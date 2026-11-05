def allocation_sweep(flops, model_sizes_b):
    """Return (parameters in billions, tokens in billions, toy loss)."""
    rows = []
    for n_b in model_sizes_b:
        d_b = tokens_at_budget(flops, n_b * BILLION) / BILLION
        loss = 1 + 0.4 / math.sqrt(n_b) + 2 / math.sqrt(d_b)
        rows.append((n_b, d_b, loss))
    return rows
