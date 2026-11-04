def build_masks(source_valid, target_valid):
    """Boolean validity arrays have shape [B, S] and [B, T]."""
    _validate_validity(source_valid, "source_valid")
    _validate_validity(target_valid, "target_valid")
    if source_valid.shape[0] != target_valid.shape[0]:
        raise ValueError("source and target batches must match")
    target_length = target_valid.shape[1]
    causal = torch.ones(
        target_length, target_length, dtype=torch.bool,
        device=target_valid.device,
    ).tril()
    source_allowed = source_valid[:, None, None, :]
    target_allowed = causal[None, None] & target_valid[:, None, None, :]
    return source_allowed, target_allowed
