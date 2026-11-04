def forward(self, x, memory, target_allowed, source_allowed):
    attended, self_weights = self.self_attention(x, x, x, target_allowed)
    x = self.norm1(x + self.self_dropout(attended))
    attended, cross_weights = self.cross_attention(
        x, memory, memory, source_allowed
    )
    x = self.norm2(x + self.cross_dropout(attended))
    x = self.norm3(x + self.ffn_dropout(self.feed_forward(x)))
    return x, self_weights, cross_weights
