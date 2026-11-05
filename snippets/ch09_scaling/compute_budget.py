def training_flops(parameters, tokens):
    """Estimate dense training FLOPs from parameter and token counts."""
    if not (math.isfinite(parameters) and parameters > 0
            and math.isfinite(tokens) and tokens > 0):
        raise ValueError("parameters and tokens must be finite and positive")
    # NumPy integer scalars can overflow at ordinary training scales.
    if isinstance(parameters, np.integer):
        parameters = int(parameters)
    if isinstance(tokens, np.integer):
        tokens = int(tokens)
    return 6 * parameters * tokens


def tokens_at_budget(flops, parameters):
    """Solve C = 6ND for D, before rounding to whole training tokens."""
    if not (math.isfinite(flops) and flops > 0):
        raise ValueError("flops must be finite and positive")
    return flops / training_flops(parameters, 1)
