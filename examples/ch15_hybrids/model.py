"""NumPy teaching examples for delta updates and per-layer history."""

import numpy as np

from examples.ch04_attention.attention import scaled_dot_product_attention


def _real_array(array, name, ndim, allow_empty=False):
    array = np.asarray(array)
    if array.ndim != ndim or (array.size == 0 and not allow_empty):
        raise ValueError(f"{name} needs {ndim} axes and a valid size")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _delta_inputs(state, query, key, value, alpha, beta):
    state = _real_array(state, "state", 2)
    query = _real_array(query, "query", 1)
    key = _real_array(key, "key", 1)
    value = _real_array(value, "value", 1)
    alpha = _real_array(alpha, "alpha", 0).item()
    beta = _real_array(beta, "beta", 0).item()
    if state.shape != (len(value), len(key)) or query.shape != key.shape:
        raise ValueError("state shape must be [value width, key and query width]")
    if not (0 <= alpha <= 1 and 0 <= beta <= 2):
        raise ValueError("alpha must lie in [0,1] and beta in [0,2]")
    return state, query, key, value, alpha, beta


# snippet:delta_step
def delta_step(state, query, key, value, alpha, beta):
    """Apply the supplied query and key vectors and return output and new state."""
    state, query, key, value, alpha, beta = _delta_inputs(
        state, query, key, value, alpha, beta
    )
    with np.errstate(over="raise", invalid="raise"):
        decayed = alpha * state
        prediction = decayed @ key
        error = value - prediction
        correction = beta * np.outer(error, key)
        updated = decayed + correction
        output = updated @ query
    return output, updated
# end-snippet:delta_step


def _attention_inputs(query, key, value, cache):
    query = _real_array(query, "query", 1)
    key = _real_array(key, "key", 1)
    value = _real_array(value, "value", 1)
    old_keys, old_values = cache
    old_keys = _real_array(old_keys, "cached keys", 2, allow_empty=True)
    old_values = _real_array(old_values, "cached values", 2, allow_empty=True)
    if old_keys.shape != (len(old_values), len(key)) or old_values.shape[1] != len(value):
        raise ValueError("cached keys and values must match length and incoming widths")
    return query, key, value, old_keys, old_values


# snippet:attention_token
def attention_token(query, key, value, cache):
    """Append one position to a full-attention cache and read the whole prefix."""
    query, key, value, old_keys, old_values = _attention_inputs(
        query, key, value, cache
    )
    keys = np.concatenate((old_keys, key[None, :]), axis=0)
    values = np.concatenate((old_values, value[None, :]), axis=0)
    output, _ = scaled_dot_product_attention(query[None, :], keys, values)
    return output[0], (keys, values)
# end-snippet:attention_token


def _schedule(layers):
    layers = tuple(layers)
    if not layers or any(layer not in ("delta", "attention") for layer in layers):
        raise ValueError("schedule needs at least one delta or attention layer")
    return layers


def initial_stores(schedule):
    """Allocate a distinct width-two state or empty KV pair for every layer."""
    return tuple(
        np.zeros((2, 2)) if mixer == "delta" else (np.empty((0, 2)), np.empty((0, 2)))
        for mixer in _schedule(schedule)
    )


def _unit_vector(vector):
    # Rescale first to keep the norm calculation safe for large features.
    scale = np.max(np.abs(vector))
    if scale == 0:
        return vector
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


# snippet:stack_step
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
# end-snippet:stack_step


# snippet:history_counts
def history_counts(stores):
    """Count the persistent state or cache scalars for each layer."""
    return tuple(
        store.size if isinstance(store, np.ndarray)
        else sum(array.size for array in store)
        for store in stores
    )
# end-snippet:history_counts
