def encode(self, source, lengths):
    """Skip right-padding and return the last valid hidden/cell pair."""
    embedded = self.source_embedding(source)
    packed = pack_padded_sequence(embedded, lengths.cpu(),
                                  batch_first=True, enforce_sorted=False)
    _, state = self.encoder(packed)
    return state

def forward(self, source, lengths, decoder_input):
    state = self.encode(source, lengths)
    embedded = self.target_embedding(decoder_input)
    hidden, _ = self.decoder(embedded, state)
    return self.output(hidden)
