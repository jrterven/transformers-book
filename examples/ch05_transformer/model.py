"""A small, batch-first, post-norm encoder-decoder Transformer.

This reference exposes heads and boolean allowed masks. It uses input and
sublayer-output dropout, independent source/target embeddings and an untied
vocabulary projection. It is not the original training recipe or a trained model.
"""

import math

import torch
from torch import nn


def _check_widths(d_model, n_heads):
    if d_model < 1 or n_heads < 1 or d_model % n_heads:
        raise ValueError("d_model must be positive and divisible by n_heads")


# snippet:positional_encoding
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
# end-snippet:positional_encoding


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        _check_widths(d_model, n_heads)
        self.n_heads, self.head_width = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.n_heads, self.head_width).transpose(1, 2)

    def merge_heads(self, x):
        batch, _, tokens, _ = x.shape
        return x.transpose(1, 2).reshape(batch, tokens, -1)

    def _validate(self, query, key, value, allowed):
        width = self.q_proj.in_features
        for name, x in (("query", query), ("key", key), ("value", value)):
            if x.ndim != 3 or x.shape[-1] != width or min(x.shape) == 0:
                raise ValueError(f"{name} must have nonempty shape [B, N, {width}]")
            if not torch.isfinite(x).all():
                raise ValueError(f"{name} must contain finite values")
        if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
            raise ValueError("batch sizes and key/value lengths must match")
        if allowed is not None:
            if allowed.dtype != torch.bool:
                raise TypeError("allowed mask must be boolean: True means allowed")
            shape = (query.shape[0], self.n_heads, query.shape[1], key.shape[1])
            try:
                allowed = torch.broadcast_to(allowed, shape)
            except RuntimeError as error:
                raise ValueError("allowed mask cannot broadcast to score shape") from error
            if not allowed.any(dim=-1).all():
                raise ValueError("each query must allow at least one key")
        return allowed

    # snippet:multihead
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
    # end-snippet:multihead


# snippet:feed_forward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.expand = nn.Linear(d_model, d_ff)
        self.contract = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.contract(torch.relu(self.expand(x)))
# end-snippet:feed_forward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

    # snippet:encoder_layer
    def forward(self, x, source_allowed):
        attended, weights = self.self_attention(x, x, x, source_allowed)
        x = self.norm1(x + self.attention_dropout(attended))
        x = self.norm2(x + self.ffn_dropout(self.feed_forward(x)))
        return x, weights
    # end-snippet:encoder_layer


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.self_dropout = nn.Dropout(dropout)
        self.cross_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)

    # snippet:decoder_layer
    def forward(self, x, memory, target_allowed, source_allowed):
        attended, self_weights = self.self_attention(x, x, x, target_allowed)
        x = self.norm1(x + self.self_dropout(attended))
        attended, cross_weights = self.cross_attention(
            x, memory, memory, source_allowed
        )
        x = self.norm2(x + self.cross_dropout(attended))
        x = self.norm3(x + self.ffn_dropout(self.feed_forward(x)))
        return x, self_weights, cross_weights
    # end-snippet:decoder_layer


def _validate_validity(valid, name):
    if valid.dtype != torch.bool or valid.ndim != 2 or min(valid.shape) == 0:
        raise ValueError(f"{name} must be a nonempty boolean [batch, tokens] array")
    if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError(f"{name} must contain a nonempty right-padded prefix")


# snippet:mask_building
def build_masks(source_valid, target_valid):
    """Boolean validity arrays have shape [B, S] and [B, T]."""
    _validate_validity(source_valid, "source_valid")
    _validate_validity(target_valid, "target_valid")
    if source_valid.shape[0] != target_valid.shape[0]:
        raise ValueError("source and target batches must match")
    target_length = target_valid.shape[1]
    causal = torch.ones(
        target_length, target_length, dtype=torch.bool,
        device=target_valid.device,
    ).tril()
    source_allowed = source_valid[:, None, None, :]
    target_allowed = causal[None, None] & target_valid[:, None, None, :]
    return source_allowed, target_allowed
# end-snippet:mask_building


class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model=4,
                 n_heads=2, d_ff=8, n_encoder_layers=1, n_decoder_layers=1,
                 dropout=0.1, pad_id=0, max_length=512):
        super().__init__()
        _check_widths(d_model, n_heads)
        if min(n_encoder_layers, n_decoder_layers) < 1:
            raise ValueError("both stacks need at least one layer")
        self.d_model, self.pad_id = d_model, pad_id
        self.source_embedding = nn.Embedding(source_vocab_size, d_model)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.position = PositionalEncoding(d_model, max_length)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.vocabulary_projection = nn.Linear(d_model, target_vocab_size)

    def _embed(self, ids, embedding):
        return self.input_dropout(self.position(embedding(ids) * math.sqrt(self.d_model)))

    def _valid_for(self, ids, valid, name):
        if ids.ndim != 2:
            raise ValueError(f"{name}_ids must have shape [batch, tokens]")
        if valid is None:
            valid = ids != self.pad_id
        if valid.shape != ids.shape:
            raise ValueError(f"{name}_valid must match token-id shape")
        return valid

    # snippet:model_forward
    def forward(self, source_ids, target_ids, source_valid=None,
                target_valid=None, return_weights=False):
        source_valid = self._valid_for(source_ids, source_valid, "source")
        target_valid = self._valid_for(target_ids, target_valid, "target")
        source_allowed, target_allowed = build_masks(source_valid, target_valid)
        memory = self._embed(source_ids, self.source_embedding)
        encoder_weights, decoder_weights, cross_weights = [], [], []
        for layer in self.encoder_layers:
            memory, weights = layer(memory, source_allowed)
            encoder_weights.append(weights)
        x = self._embed(target_ids, self.target_embedding)
        for layer in self.decoder_layers:
            x, self_weights, cross = layer(x, memory, target_allowed, source_allowed)
            decoder_weights.append(self_weights)
            cross_weights.append(cross)
        logits = self.vocabulary_projection(x)
        if return_weights:
            return logits, {"encoder": encoder_weights,
                            "decoder": decoder_weights, "cross": cross_weights}
        return logits
    # end-snippet:model_forward
