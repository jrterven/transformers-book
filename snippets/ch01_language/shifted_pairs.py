def next_token_pairs(token_ids):
    """Use two BOS tokens to fill missing context at the start of a sentence."""
    if len(token_ids) == 0:
        raise ValueError("a sentence must contain at least one word")
    token_ids = _check_ids(token_ids, len(VOCAB))
    if token_ids.ndim != 1 or np.isin(token_ids, (PAD, BOS, EOS)).any():
        raise ValueError("sentence IDs must be a one-dimensional list of word IDs")
    padded = np.concatenate(([BOS, BOS], token_ids, [EOS]))
    contexts = np.stack([padded[t:t + 2] for t in range(len(token_ids) + 1)])
    targets = padded[2:]
    return contexts, targets
