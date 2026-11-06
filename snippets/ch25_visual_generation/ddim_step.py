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
