"""NumPy teaching examples for delta updates and per-layer history."""

from operator import index

import numpy as np

from examples.ch04_attention.attention import scaled_dot_product_attention
from examples.ch14_selective_ssm.model import causal_decay


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


def _delta_sequence_inputs(query, key, value, alpha, beta, state):
    query = _real_array(query, "queries", 2)
    key = _real_array(key, "keys", 2)
    value = _real_array(value, "values", 2)
    alpha = _real_array(alpha, "alpha", 1)
    beta = _real_array(beta, "beta", 1)
    if (query.shape != key.shape or len(value) != len(key)
            or alpha.shape != (len(key),) or beta.shape != (len(key),)):
        raise ValueError("all sequence arrays must have the same time axis")
    if np.any((alpha < 0) | (alpha > 1)) or np.any((beta < 0) | (beta > 2)):
        raise ValueError("alpha must lie in [0,1] and beta in [0,2]")
    if state is None:
        state = np.zeros((value.shape[1], key.shape[1]))
    state = _real_array(state, "initial state", 2)
    if state.shape != (value.shape[1], key.shape[1]):
        raise ValueError("state shape must be [value width, key width]")
    return query, key, value, alpha, beta, state


# snippet:delta_sequence
def delta_sequence(query, key, value, alpha, beta, state=None):
    """Evaluate a sequence; return its outputs and final [value, key] state."""
    query, key, value, alpha, beta, state = _delta_sequence_inputs(
        query, key, value, alpha, beta, state
    )
    outputs = np.empty_like(value)
    for t in range(len(key)):
        outputs[t], state = delta_step(
            state, query[t], key[t], value[t], alpha[t], beta[t]
        )
    return outputs, state
# end-snippet:delta_sequence


# snippet:delta_chunked
def delta_chunked(query, key, value, alpha, beta, chunk_size, state=None):
    """Solve local error dependencies and pass only the final matrix state."""
    query, key, value, alpha, beta, state = _delta_sequence_inputs(
        query, key, value, alpha, beta, state
    )
    chunk_size = index(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    outputs = np.empty_like(value)
    with np.errstate(over="raise", invalid="raise"):
        for start in range(0, len(key), chunk_size):
            rows = slice(start, start + chunk_size)
            q, k, v, a, b = (x[rows] for x in (query, key, value, alpha, beta))
            decay = causal_decay(a)
            prefix = np.cumprod(a)
            dependencies = decay * (k @ k.T) * b[None, :]
            system = np.eye(len(k)) + np.tril(dependencies, k=-1)
            residual = v - prefix[:, None] * (k @ state.T)
            errors = np.linalg.solve(system, residual)
            reads = decay * (q @ k.T) * b[None, :]
            outputs[rows] = prefix[:, None] * (q @ state.T) + reads @ errors
            writes = (errors * (decay[-1] * b)[:, None]).T @ k
            state = prefix[-1] * state + writes
    return outputs, state
# end-snippet:delta_chunked


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
