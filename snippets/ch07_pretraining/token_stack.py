def forward(self, ids):
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
    return self.vocabulary_projection(x)     # [B, T, V]
