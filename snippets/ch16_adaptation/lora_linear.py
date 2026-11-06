class LoRALinear(nn.Module):
    """Keep a Linear base frozen and train a low-rank weight update."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        rank = index(rank)
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be a torch.nn.Linear layer")
        if not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError("rank must fit both weight dimensions")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / rank
        self.a = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.b = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a), self.b)

    def merged_weight(self):
        return self.base.weight + self.scale * (self.b @ self.a)
