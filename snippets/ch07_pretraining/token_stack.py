def encode_hidden(self, ids):
    """Return one final hidden vector per input position, before a head."""
    valid = ids != PAD                       # [B, T]
    _check_right_padding(valid)
    allowed = valid[:, None, None, :]         # available keys
    if self.causal:
        length = ids.shape[1]
        triangle = torch.ones(length, length, dtype=torch.bool,
                              device=ids.device).tril()
        allowed = allowed & triangle[None, None]
    x = self.position(self.embedding(ids) * math.sqrt(self.d_model))
    for layer in self.layers:
        x, _ = layer(x, allowed)
    return x                                # [B, T, d_model]

def forward(self, ids):
    return self.vocabulary_projection(self.encode_hidden(ids))  # [B, T, V]
