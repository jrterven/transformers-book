def stack_step(token, schedule, stores):
    """Return this token's layer trace and the new per-layer history stores."""
    schedule = _schedule(schedule)
    h = _real_array(token, "token", 1)
    if h.shape != (2,) or len(stores) != len(schedule):
        raise ValueError("use a width-two token and one store per layer")

    representations = [h]
    next_stores = []
    for mixer, store in zip(schedule, stores):
        # Identity query and key projections; the value projection swaps features.
        value = h[::-1]
        if mixer == "delta":
            direction = _unit_vector(h)
            h, updated_store = delta_step(
                store, direction, direction, value, alpha=0.75, beta=0.5
            )
        else:
            h, updated_store = attention_token(h, h, value, store)
        representations.append(h)
        next_stores.append(updated_store)
    return tuple(representations), tuple(next_stores)
