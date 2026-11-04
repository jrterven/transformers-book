def token_loss(logits, target):
    """Average over non-padding labels, including each sequence's EOS."""
    if torch is None:
        raise ImportError("Install requirements-ch02.txt for the PyTorch example")
    if not (target != 0).any():
        raise ValueError("at least one non-padding target is required")
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
        ignore_index=0,
    )
