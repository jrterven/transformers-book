"""NumPy teaching examples for fixed-coefficient state space models."""

from operator import index

import numpy as np


def _real_array(array, name, ndim):
    array = np.asarray(array)
    if array.ndim != ndim or array.size == 0:
        shape_name = ("scalar", "vector", "matrix")[ndim]
        raise ValueError(f"{name} must be a nonempty {shape_name}")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _transition_and_input(a, b):
    a = _real_array(a, "A", 2)
    b = _real_array(b, "B", 1)
    if a.shape != (len(b), len(b)):
        raise ValueError("A must be square with width matching B")
    return a, b


def _state_space(a_bar, b_bar, c):
    a_bar, b_bar = _transition_and_input(a_bar, b_bar)
    c = _real_array(c, "C", 1)
    if c.shape != b_bar.shape:
        raise ValueError("B and C must match the state width")
    return a_bar, b_bar, c


# snippet:ssm_scan
def ssm_scan(u, a_bar, b_bar, c, x0=None):
    """Return outputs [time] and the states after each update [time, width]."""
    u = _real_array(u, "u", 1)
    a_bar, b_bar, c = _state_space(a_bar, b_bar, c)
    state = np.zeros_like(b_bar) if x0 is None else _real_array(x0, "x0", 1)
    if state.shape != b_bar.shape:
        raise ValueError("x0 must match the state width")

    outputs = np.empty(len(u))
    states = np.empty((len(u), len(state)))
    with np.errstate(over="raise", invalid="raise"):
        for t, token in enumerate(u):
            state = a_bar @ state + b_bar * token
            states[t] = state
            outputs[t] = c @ state
    return outputs, states
# end-snippet:ssm_scan


# snippet:ssm_kernel
def ssm_kernel(a_bar, b_bar, c, length):
    """Read the response to a unit impulse at lags zero through length - 1."""
    a_bar, b_bar, c = _state_space(a_bar, b_bar, c)
    length = index(length)
    if length < 1:
        raise ValueError("kernel length must be positive")

    impulse = b_bar
    kernel = np.empty(length)
    with np.errstate(over="raise", invalid="raise"):
        kernel[0] = c @ impulse
        for lag in range(1, length):
            impulse = a_bar @ impulse
            kernel[lag] = c @ impulse
    return kernel
# end-snippet:ssm_kernel


# snippet:causal_convolution
def causal_convolution(u, kernel):
    """Sum the available lag terms at each input position."""
    u = _real_array(u, "u", 1)
    kernel = _real_array(kernel, "kernel", 1)
    output = np.empty(len(u))
    with np.errstate(over="raise", invalid="raise"):
        for t in range(len(u)):
            count = min(t + 1, len(kernel))
            newest_first = u[t - count + 1:t + 1][::-1]
            output[t] = kernel[:count] @ newest_first
    return output


def fft_causal_convolution(u, kernel):
    """Use a padded real FFT, then keep the first len(u) positions."""
    u = _real_array(u, "u", 1)
    kernel = _real_array(kernel, "kernel", 1)
    fft_length = len(u) + len(kernel) - 1
    with np.errstate(over="raise", invalid="raise"):
        spectrum = np.fft.rfft(u, n=fft_length) * np.fft.rfft(kernel, n=fft_length)
        full = np.fft.irfft(spectrum, n=fft_length)
    return full[:len(u)]
# end-snippet:causal_convolution


# snippet:bilinear_discretize
def bilinear_discretize(a, b, delta):
    """Compute the discrete transition and input map of a continuous system."""
    a, b = _transition_and_input(a, b)
    delta = _real_array(delta, "delta", 0).item()
    if delta <= 0:
        raise ValueError("delta must be positive")

    identity = np.eye(len(b))
    with np.errstate(over="raise", invalid="raise"):
        half_step = (delta / 2) * a
        left = identity - half_step
        a_bar = np.linalg.solve(left, identity + half_step)
        b_bar = np.linalg.solve(left, delta * b)
    if not (np.isfinite(a_bar).all() and np.isfinite(b_bar).all()):
        raise FloatingPointError("discretization produced a nonfinite result")
    return a_bar, b_bar
# end-snippet:bilinear_discretize


# snippet:hyena_stage
def hyena_stage(u, input_gate, output_gate, kernel):
    """Return the gated input, filtered sequence, and gated output."""
    u = _real_array(u, "u", 1)
    input_gate = _real_array(input_gate, "input gate", 1)
    output_gate = _real_array(output_gate, "output gate", 1)
    if not (u.shape == input_gate.shape == output_gate.shape):
        raise ValueError("each gate needs one entry per input position")

    with np.errstate(over="raise", invalid="raise"):
        gated_input = input_gate * u
        filtered = causal_convolution(gated_input, kernel)
        output = output_gate * filtered
    return gated_input, filtered, output
# end-snippet:hyena_stage
