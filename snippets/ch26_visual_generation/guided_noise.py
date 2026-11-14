def guided_noise(unconditional, conditional, scale):
    """Combine two epsilon predictions with one scalar guidance scale."""
    unconditional, conditional = _same_shape(
        unconditional, conditional, "unconditional", "conditional"
    )
    scale = _real_scalar(scale, "scale")
    with np.errstate(over="raise", invalid="raise"):
        return unconditional + scale * (conditional - unconditional)
