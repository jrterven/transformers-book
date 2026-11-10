@torch.no_grad()
def merge_lora(model, targets):
    """Replace selected adapters with frozen merged Linear layers for inference."""
    targets = tuple(targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("targets must be nonempty unique module paths")
    layers = [model.get_submodule(name) for name in targets]
    if any(not isinstance(layer, LoRALinear) for layer in layers):
        raise TypeError("each merge target must be a LoRALinear module")
    for name, layer in zip(targets, layers):
        base = layer.base
        merged = nn.Linear(base.in_features, base.out_features,
                           bias=base.bias is not None,
                           dtype=base.weight.dtype, device=base.weight.device)
        merged.weight.copy_(layer.merged_weight())
        if base.bias is not None:
            merged.bias.copy_(base.bias)
        model.set_submodule(name, merged.requires_grad_(False))
