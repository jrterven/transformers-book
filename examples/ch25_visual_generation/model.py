"""NumPy teaching equations for epsilon-parameterized diffusion."""

import math

import numpy as np


def _real_array(values, name):
    values = np.asarray(values)
    if values.size == 0 or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be nonempty and real")
    with np.errstate(over="raise", invalid="raise"):
        values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _same_shape(first, second, first_name, second_name):
    first = _real_array(first, first_name)
    second = _real_array(second, second_name)
    if first.shape != second.shape:
        raise ValueError(f"{first_name} and {second_name} must have the same shape")
    return first, second


def _real_scalar(value, name):
    value = _real_array(value, name)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return value.item()


# snippet:add_noise
def add_noise(clean, noise, alpha_bar):
    """Evaluate one forward marginal with the supplied noise sample."""
    clean, noise = _same_shape(clean, noise, "clean", "noise")
    alpha_bar = _real_scalar(alpha_bar, "alpha_bar")
    if not 0 <= alpha_bar <= 1:
        raise ValueError("alpha_bar must lie between zero and one")
    with np.errstate(over="raise", invalid="raise"):
        return math.sqrt(alpha_bar) * clean + math.sqrt(1 - alpha_bar) * noise
# end-snippet:add_noise


# snippet:guided_noise
def guided_noise(unconditional, conditional, scale):
    """Combine two epsilon predictions with one scalar guidance scale."""
    unconditional, conditional = _same_shape(
        unconditional, conditional, "unconditional", "conditional"
    )
    scale = _real_scalar(scale, "scale")
    with np.errstate(over="raise", invalid="raise"):
        return unconditional + scale * (conditional - unconditional)
# end-snippet:guided_noise


# snippet:ddim_step
def ddim_step(noisy, predicted_noise, alpha_bar_t, alpha_bar_next):
    """Return the estimated clean sample and the next deterministic sample."""
    noisy, predicted_noise = _same_shape(
        noisy, predicted_noise, "noisy", "predicted_noise"
    )
    alpha_bar_t = _real_scalar(alpha_bar_t, "alpha_bar_t")
    alpha_bar_next = _real_scalar(alpha_bar_next, "alpha_bar_next")
    if not 0 < alpha_bar_t <= alpha_bar_next <= 1:
        raise ValueError("use 0 < alpha_bar_t <= alpha_bar_next <= 1")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        estimated_clean = (
            noisy - math.sqrt(1 - alpha_bar_t) * predicted_noise
        ) / math.sqrt(alpha_bar_t)
        next_noisy = (math.sqrt(alpha_bar_next) * estimated_clean
                      + math.sqrt(1 - alpha_bar_next) * predicted_noise)
    return estimated_clean, next_noisy
# end-snippet:ddim_step
