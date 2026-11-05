def selected_loss(logits, labels):
    if logits.shape[:-1] != labels.shape:
        raise ValueError("each input position must have one label slot")
    if not (labels != IGNORE).any():
        raise ValueError("select at least one label before calculating a mean")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=IGNORE,
    )
