def attach_lora(model, targets, rank, alpha):
    """Freeze the model and add adapters at explicit Linear module paths."""
    targets, rank = tuple(targets), index(rank)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("targets must be nonempty unique module paths")
    bases = [model.get_submodule(name) for name in targets]
    if any(not isinstance(base, nn.Linear) for base in bases):
        raise TypeError("each target must be a Linear module")
    if (not math.isfinite(alpha) or alpha <= 0 or rank < 1
            or any(rank > min(base.in_features, base.out_features) for base in bases)):
        raise ValueError("rank must fit every target and alpha must be positive")
    model.requires_grad_(False)
    adapters = {}
    for name, base in zip(targets, bases):
        adapters[name] = LoRALinear(base, rank, alpha)
        model.set_submodule(name, adapters[name])
    return adapters
