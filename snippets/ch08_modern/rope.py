def rotate_pairs(x, positions, base=10000.0):
    """Rotate adjacent feature pairs in [B, H, T, head_width]."""
    if x.ndim != 4 or min(x.shape) == 0 or x.shape[-1] % 2:
        raise ValueError("x needs nonempty shape [B, H, T, even head_width]")
    if not x.is_floating_point():
        raise TypeError("x must be floating point")
    if positions.shape != (x.shape[-2],):
        raise ValueError("positions must have one index per token")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must be integers")
    if not math.isfinite(base) or base <= 0:
        raise ValueError("base must be positive and finite")

    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    width = x.shape[-1]
    pairs = torch.arange(0, width, 2, device=x.device, dtype=dtype)
    frequencies = base ** (-pairs / width)
    angles = positions.to(device=x.device, dtype=dtype)[:, None] * frequencies
    cosine, sine = angles.cos(), angles.sin()
    even, odd = x[..., ::2].to(dtype), x[..., 1::2].to(dtype)
    rotated = torch.stack((even * cosine - odd * sine,
                           even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2).to(dtype=x.dtype)
