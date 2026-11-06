def response_labels(ids, response_mask, pad_id=PAD):
    """Use a mask on unshifted tokens to select next-token targets."""
    pad_id = index(pad_id)
    if pad_id < 0:
        raise ValueError("pad_id must be nonnegative")
    _response_inputs(ids, response_mask, pad_id)
    targets = ids[:, 1:]
    selected = response_mask[:, 1:] & targets.ne(pad_id)
    if not selected.any(dim=1).all():
        raise ValueError("each row needs at least one response target")
    return targets.masked_fill(~selected, IGNORE)
