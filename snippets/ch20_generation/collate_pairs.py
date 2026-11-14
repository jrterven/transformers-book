def collate_pairs(pairs, *, pad_id=PAD, bos_id=BOS, eos_id=EOS):
    """Return source IDs, shifted decoder input IDs, and target labels."""
    pad_id, bos_id, eos_id = map(_token_id, (pad_id, bos_id, eos_id))
    special = {pad_id, bos_id, eos_id}
    if len(special) != 3:
        raise ValueError("PAD, BOS, and EOS must have distinct IDs")

    sources, decoder_inputs, labels = [], [], []
    for source, target in pairs:
        source = list(map(_token_id, source))
        target = list(map(_token_id, target))
        if not source or pad_id in source:
            raise ValueError("each source must be nonempty and unpadded")
        if any(token in special for token in target):
            raise ValueError("target content must exclude PAD, BOS, and EOS")
        sources.append(source)
        decoder_inputs.append([bos_id] + target)
        labels.append(target + [eos_id])
    if not sources:
        raise ValueError("a batch needs at least one source-target pair")
    return tuple(_right_pad(rows, pad_id) for rows in (sources, decoder_inputs, labels))
