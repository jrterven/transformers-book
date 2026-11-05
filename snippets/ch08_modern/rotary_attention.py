def forward(self, x, allowed, positions):
    allowed = self._validate(x, x, x, allowed)
    q = rotate_pairs(self.split_heads(self.q_proj(x)), positions)
    k = rotate_pairs(self.split_heads(self.k_proj(x)), positions)
    v = self.split_heads(self.v_proj(x))
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
    if not torch.isfinite(scores).all():
        raise ValueError("attention scores must be finite before masking")
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return self.out_proj(self.merge_heads(weights @ v)), weights
