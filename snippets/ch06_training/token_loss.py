def token_loss(logits, targets):
    count = (targets != PAD).sum()
    if count.item() == 0:
        raise ValueError("a loss needs at least one non-padding target")
    total = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=PAD, reduction="sum",
    )
    return total, count
