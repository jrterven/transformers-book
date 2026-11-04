def collate(sequences):
    """BOS goes into the decoder; EOS is one of the labels it must predict."""
    source = [list(row) for row in sequences]
    if not source or any(not row or any(t not in COLORS for t in row)
                         for row in source):
        raise ValueError("sources must be nonempty lists of color IDs")
    reversed_rows = [list(reversed(row)) for row in source]
    decoder_input = [[BOS] + row for row in reversed_rows]
    target = [row + [EOS] for row in reversed_rows]
    return Batch(pad(source), torch.tensor([len(row) for row in source]),
                 pad(decoder_input), pad(target))
