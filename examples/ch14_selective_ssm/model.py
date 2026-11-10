"""NumPy teaching examples for selective SSMs and scalar-decay SSD."""

from operator import index

import numpy as np


def _real_array(array, name, ndim=None):
    array = np.asarray(array)
    if array.size == 0 or (ndim is not None and array.ndim != ndim):
        shape = "a nonempty array" if ndim is None else f"a nonempty {ndim}D array"
        raise ValueError(f"{name} must be {shape}")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _retentions(a, ndim):
    a = _real_array(a, "retentions", ndim)
    if np.any((a < 0) | (a > 1)):
        raise ValueError("retentions must lie between zero and one")
    return a


def _sequence_inputs(u, b, c, x0):
    u = _real_array(u, "u", 1)
    b = _real_array(b, "b", 2)
    c = _real_array(c, "c", 2)
    if b.shape != c.shape or len(u) != len(b):
        raise ValueError("b and c must share shape [time, state] and match u")
    state = np.zeros(b.shape[1]) if x0 is None else _real_array(x0, "x0", 1)
    if state.shape != (b.shape[1],):
        raise ValueError("x0 must match the state width")
    return u, b, c, state


def _ssd_retentions(a, length):
    a = _retentions(a, 1)
    if len(a) != length:
        raise ValueError("SSD needs one retention per time step")
    return a


# snippet:zoh_coefficients
def zoh_coefficients(delta):
    """Return retain and write coefficients for dx/dt = -x + u."""
    delta = _real_array(delta, "delta")
    if np.any(delta <= 0):
        raise ValueError("steps must be positive")
    with np.errstate(under="ignore"):
        retain = np.exp(-delta)
        write = -np.expm1(-delta)
    return retain, write
# end-snippet:zoh_coefficients


# snippet:selective_scan
def selective_scan(u, a, b, c, x0=None):
    """Return outputs and states for a precomputed diagonal update."""
    u, b, c, state = _sequence_inputs(u, b, c, x0)
    a = _retentions(a, 2)
    if a.shape not in (b.shape, (len(u), 1)):
        raise ValueError("a must have one or state-width retentions per time step")

    outputs = np.empty(len(u))
    states = np.empty_like(b)
    with np.errstate(over="raise", invalid="raise"):
        for t, token in enumerate(u):
            state = a[t] * state + b[t] * token
            states[t] = state
            outputs[t] = c[t] @ state
    return outputs, states
# end-snippet:selective_scan


# snippet:compose_diagonal
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
# end-snippet:compose_diagonal


# snippet:ssd_dense
def causal_decay(a):
    """Build L[t, s] as the product of retentions after write s through t."""
    a = _retentions(a, 1)
    decay = np.eye(len(a))
    for t in range(1, len(a)):
        decay[t, :t] = a[t] * decay[t - 1, :t]
    return decay


def ssd_dense(u, a, b, c, x0=None):
    """Return outputs and the full matrix mapping inputs to outputs."""
    u, b, c, state = _sequence_inputs(u, b, c, x0)
    a = _ssd_retentions(a, len(u))
    with np.errstate(over="raise", invalid="raise"):
        matrix = causal_decay(a) * (c @ b.T)
        from_x0 = np.cumprod(a) * (c @ state)
        outputs = matrix @ u + from_x0
    return outputs, matrix
# end-snippet:ssd_dense


def ssd_recurrent(u, a, b, c, x0=None):
    """Broadcast each scalar retention over all state coordinates."""
    a = _real_array(a, "retentions", 1)
    return selective_scan(u, a[:, None], b, c, x0)


# snippet:ssd_chunked
def ssd_chunked(u, a, b, c, chunk_size, x0=None):
    """Return outputs and the last state, building only local decay blocks."""
    u, b, c, state = _sequence_inputs(u, b, c, x0)
    a = _ssd_retentions(a, len(u))
    chunk_size = index(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")

    outputs = np.empty(len(u))
    with np.errstate(over="raise", invalid="raise"):
        for start in range(0, len(u), chunk_size):
            rows = slice(start, start + chunk_size)
            u_block, b_block, c_block = u[rows], b[rows], c[rows]
            decay = causal_decay(a[rows])
            prefix = np.cumprod(a[rows])
            within = (decay * (c_block @ b_block.T)) @ u_block
            outputs[rows] = within + prefix * (c_block @ state)

            # The final row gathers all writes ending at this chunk boundary.
            last_write = (decay[-1] * u_block) @ b_block
            state = prefix[-1] * state + last_write
    return outputs, state
# end-snippet:ssd_chunked


def _head_inputs(u, a, b, c, x0):
    u = _real_array(u, "u", 2)
    b = _real_array(b, "b", 2)
    c = _real_array(c, "c", 2)
    if b.shape != c.shape or len(u) != len(b):
        raise ValueError("b and c must share shape [time, state] and match u")
    a = _ssd_retentions(a, len(u))
    shape = (b.shape[1], u.shape[1])
    state = np.zeros(shape) if x0 is None else _real_array(x0, "x0", 2)
    if state.shape != shape:
        raise ValueError("x0 must have shape [state, feature]")
    return u, a, b, c, state


def ssd_head_recurrent(u, a, b, c, x0=None):
    """Run a width-P SSD head with state [N, P], saving its trajectory."""
    u, a, b, c, state = _head_inputs(u, a, b, c, x0)
    outputs = np.empty_like(u)
    states = np.empty((len(u), *state.shape))
    with np.errstate(over="raise", invalid="raise"):
        for t in range(len(u)):
            state = a[t] * state + np.outer(b[t], u[t])
            states[t] = state
            outputs[t] = c[t] @ state
    return outputs, states


# snippet:ssd_head_chunked
def ssd_head_chunked(u, a, b, c, chunk_size, x0=None):
    """Evaluate a width-P SSD head; carry one [N, P] boundary state."""
    u, a, b, c, state = _head_inputs(u, a, b, c, x0)
    chunk_size = index(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    outputs = np.empty_like(u)
    with np.errstate(over="raise", invalid="raise"):
        for start in range(0, len(u), chunk_size):
            rows = slice(start, start + chunk_size)
            ub, bb, cb = u[rows], b[rows], c[rows]
            decay = causal_decay(a[rows])
            prefix = np.cumprod(a[rows])
            within = (decay * (cb @ bb.T)) @ ub
            outputs[rows] = within + prefix[:, None] * (cb @ state)
            last_write = bb.T @ (decay[-1, :, None] * ub)
            state = prefix[-1] * state + last_write
    return outputs, state
# end-snippet:ssd_head_chunked
