def add_noise(clean, noise, alpha_bar):
    """Evaluate one forward marginal with the supplied noise sample."""
    clean, noise = _same_shape(clean, noise, "clean", "noise")
    alpha_bar = _real_scalar(alpha_bar, "alpha_bar")
    if not 0 <= alpha_bar <= 1:
        raise ValueError("alpha_bar must lie between zero and one")
    with np.errstate(over="raise", invalid="raise"):
        return math.sqrt(alpha_bar) * clean + math.sqrt(1 - alpha_bar) * noise
