def forward(self, query, key, value, allowed=None):
    allowed = self._validate(query, key, value, allowed)
    q = self.split_heads(self.q_proj(query))
    k = self.split_heads(self.k_proj(key))
    v = self.split_heads(self.v_proj(value))
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
    if not torch.isfinite(scores).all():
        raise ValueError("attention scores must be finite before masking")
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    heads = weights @ v
    combined = self.merge_heads(heads)
    return self.out_proj(combined), weights
