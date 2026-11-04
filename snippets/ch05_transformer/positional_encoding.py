class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=512):
        super().__init__()
        if d_model < 1 or max_length < 1:
            raise ValueError("d_model and max_length must be positive")
        position = torch.arange(max_length, dtype=torch.float64)[:, None]
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float64)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_length, d_model, dtype=torch.float64)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency[:d_model // 2])
        self.register_buffer("pe", pe[None, :, :])

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.pe.shape[-1]:
            raise ValueError("x must have shape [batch, tokens, d_model]")
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError("sequence exceeds max_length")
        return x + self.pe[:, :x.shape[1]].to(dtype=x.dtype)
