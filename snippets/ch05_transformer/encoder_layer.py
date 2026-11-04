def forward(self, x, source_allowed):
    attended, weights = self.self_attention(x, x, x, source_allowed)
    x = self.norm1(x + self.attention_dropout(attended))
    x = self.norm2(x + self.ffn_dropout(self.feed_forward(x)))
    return x, weights
