"""Educational information paths, rather than GPT, BERT, or T5 replicas.

The shared small block deliberately keeps Chapter 5's sinusoidal positions,
ReLU feed-forward network, and post-norm residual paths. It does not implement
the original families' learned positions, GELU/MLM heads, relative biases, or
other model-specific details. All weights start random.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from examples.ch05_transformer.model import EncoderLayer, PositionalEncoding, Transformer
from .data import PAD, IGNORE, VOCAB


class TinyTokenModel(nn.Module):
    def __init__(self, causal, d_model=16, n_heads=2, d_ff=32,
                 n_layers=2, max_length=32):
        super().__init__()
        if n_layers < 1:
            raise ValueError("at least one layer is required")
        self.causal = causal
        self.d_model = d_model
        self.embedding = nn.Embedding(len(VOCAB), d_model)
        self.position = PositionalEncoding(d_model, max_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout=0.0)
            for _ in range(n_layers)
        ])
        self.vocabulary_projection = nn.Linear(d_model, len(VOCAB))

    # snippet:token_stack
    def encode_hidden(self, ids):
        """Return one final hidden vector per input position, before a head."""
        valid = ids != PAD                       # [B, T]
        _check_right_padding(valid)
        allowed = valid[:, None, None, :]         # available keys
        if self.causal:
            length = ids.shape[1]
            triangle = torch.ones(length, length, dtype=torch.bool,
                                  device=ids.device).tril()
            allowed = allowed & triangle[None, None]
        x = self.position(self.embedding(ids) * math.sqrt(self.d_model))
        for layer in self.layers:
            x, _ = layer(x, allowed)
        return x                                # [B, T, d_model]

    def forward(self, ids):
        return self.vocabulary_projection(self.encode_hidden(ids))  # [B, T, V]
    # end-snippet:token_stack


def _check_right_padding(valid):
    if valid.ndim != 2 or min(valid.shape) == 0:
        raise ValueError("ids must have nonempty shape [batch, tokens]")
    if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("each row needs a nonempty right-padded prefix")


# snippet:selected_loss
def selected_loss(logits, labels):
    if logits.shape[:-1] != labels.shape:
        raise ValueError("each input position must have one label slot")
    if not (labels != IGNORE).any():
        raise ValueError("select at least one label before calculating a mean")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=IGNORE,
    )
# end-snippet:selected_loss


def make_model(family, d_model=16, n_heads=2, d_ff=32, n_layers=2):
    if family in ("gpt", "bert"):
        return TinyTokenModel(family == "gpt", d_model, n_heads, d_ff, n_layers)
    if family == "t5":
        return Transformer(len(VOCAB), len(VOCAB), d_model=d_model,
                           n_heads=n_heads, d_ff=d_ff,
                           n_encoder_layers=n_layers, n_decoder_layers=n_layers,
                           dropout=0.0, pad_id=PAD, max_length=32)
    raise ValueError("family must be gpt, bert, or t5")
