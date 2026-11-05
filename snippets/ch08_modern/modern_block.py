class ModernBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = RotarySelfAttention(d_model, n_heads)
        self.feed_forward_norm = RMSNorm(d_model)
        self.feed_forward = SwiGLU(d_model, d_ff)

    def forward(self, x, allowed, positions):
        attended, _ = self.attention(self.attention_norm(x), allowed, positions)
        x = x + attended
        return x + self.feed_forward(self.feed_forward_norm(x))
