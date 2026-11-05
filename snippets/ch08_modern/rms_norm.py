def rms_norm(x, weight, eps=1e-6):
    """Scale the feature vector at each token position."""
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    mean_square = work.square().mean(dim=-1, keepdim=True)
    normalized = work * torch.rsqrt(mean_square + eps)
    return (normalized * weight).to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        if width < 1 or not math.isfinite(eps) or eps < 0:
            raise ValueError("width must be positive and eps nonnegative")
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)
