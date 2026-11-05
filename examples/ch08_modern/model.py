"""A teaching decoder with pre-RMSNorm, SwiGLU, and rotary attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from examples.ch05_transformer.model import MultiHeadAttention, build_masks


# snippet:rms_norm
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
# end-snippet:rms_norm


# snippet:swiglu
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        features = self.up_proj(x)
        return self.down_proj(gate * features)
# end-snippet:swiglu


# snippet:rope
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
# end-snippet:rope


class RotarySelfAttention(MultiHeadAttention):
    """Reuse the projections, head layout, and mask checks from Chapter 5."""

    def __init__(self, d_model, n_heads):
        super().__init__(d_model, n_heads)
        if self.head_width % 2:
            raise ValueError("rotary attention needs an even head width")

    # snippet:rotary_attention
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
    # end-snippet:rotary_attention


# snippet:modern_block
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
# end-snippet:modern_block


class ModernDecoder(nn.Module):
    """A small decoder-only stack with an untied vocabulary projection."""

    def __init__(self, vocab_size=17, d_model=16, n_heads=2, d_ff=32,
                 n_layers=2, pad_id=0):
        super().__init__()
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id must be in the vocabulary")
        if n_layers < 1:
            raise ValueError("the decoder needs at least one layer")
        if d_model < 1 or d_ff < 1:
            raise ValueError("model and intermediate widths must be positive")
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            ModernBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.vocabulary_projection = nn.Linear(d_model, vocab_size, bias=False)

    # snippet:modern_decoder
    def forward(self, ids, start_pos=0):
        if ids.ndim != 2:
            raise ValueError("ids must have shape [batch, tokens]")
        if not isinstance(start_pos, int) or start_pos < 0:
            raise ValueError("start_pos must be a nonnegative integer")
        valid = ids != self.pad_id
        _, allowed = build_masks(valid, valid)
        positions = torch.arange(start_pos, start_pos + ids.shape[1],
                                 device=ids.device)
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, allowed, positions)
        return self.vocabulary_projection(self.final_norm(x))
    # end-snippet:modern_decoder
