def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    """Compare policy preference margins relative to a frozen reference."""
    pairs = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    if (policy_chosen.numel() == 0
            or any(x.shape != policy_chosen.shape or not x.is_floating_point() for x in pairs)
            or any(not torch.isfinite(x).all() for x in pairs)):
        raise ValueError("log-probabilities must have matching shapes and be finite")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")

    policy_chosen, policy_rejected, reference_chosen, reference_rejected = (
        x.float() if x.dtype in (torch.float16, torch.bfloat16) else x for x in pairs
    )
    policy_gap = policy_chosen - policy_rejected
    reference_gap = reference_chosen.detach() - reference_rejected.detach()
    margin = beta * (policy_gap - reference_gap)
    if not torch.isfinite(margin).all():
        raise FloatingPointError("DPO margin overflowed")
    return F.softplus(-margin).mean()
