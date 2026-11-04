def collate(sequences):
    source = [list(sequence) for sequence in sequences]
    if not source or any(not row or any(t not in COLORS for t in row)
                         for row in source):
        raise ValueError("each source must be a nonempty sequence of color IDs")
    targets = [list(reversed(row)) for row in source]
    decoder_input = [[BOS] + row for row in targets]
    labels = [row + [EOS] for row in targets]
    return Batch(_pad(source), _pad(decoder_input), _pad(labels))
