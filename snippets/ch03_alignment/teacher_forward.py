def forward(self, source, lengths, decoder_input):
    memory, state = self.encode(source, lengths)
    if decoder_input.ndim != 2 or decoder_input.shape[1] == 0:
        raise ValueError("decoder inputs need at least one position")
    logits, alignments = [], []
    for previous in decoder_input.unbind(dim=1):
        scores, state, weights = self.step(previous, state, memory)
        logits.append(scores)
        alignments.append(weights)
    return torch.stack(logits, dim=1), torch.stack(alignments, dim=1)
