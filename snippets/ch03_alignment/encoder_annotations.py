def encode(self, source, lengths):
    if (source.ndim != 2 or lengths.shape != (len(source),)
            or len(source) == 0 or (lengths < 1).any()
            or (lengths > source.shape[1]).any()):
        raise ValueError("provide a nonempty source batch and valid lengths")
    valid = torch.arange(source.shape[1], device=source.device)[None, :]
    valid = valid < lengths.to(source.device)[:, None]
    if (source[valid] == 0).any():
        raise ValueError("valid source positions must not be PAD")
    packed = pack_padded_sequence(self.source_embedding(source), lengths.cpu(),
                                  batch_first=True, enforce_sorted=False)
    packed_states, (hidden, cell) = self.encoder(packed)
    annotations, _ = pad_packed_sequence(
        packed_states, batch_first=True, total_length=source.shape[1])
    projected = self.attention.annotation(annotations)
    return (annotations, projected, valid), (hidden[0], cell[0])
