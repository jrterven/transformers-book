"""Full-prefix and incremental decoding with the same Chapter 5 weights.

The cache is deliberately transparent: torch.cat grows each layer's self-KV.
This is a reference, not an optimized inference engine. Generation runs in
evaluation mode and never supplies a reference target token.
"""

import math
from time import perf_counter
import statistics

import torch

from examples.ch05_transformer.model import build_masks
from .data import PAD, BOS, EOS


def _require_eval(model):
    if model.training:
        raise ValueError("call model.eval() before generation or cache use")


@torch.no_grad()
def encode_source(model, source):
    _require_eval(model)
    valid = source != model.pad_id
    # Reuse Chapter 5's mask validation, including its right-padding contract.
    source_allowed, _ = build_masks(valid, valid[:, :1])
    memory = model._embed(source, model.source_embedding)
    for layer in model.encoder_layers:
        memory, _ = layer(memory, source_allowed)
    return memory, source_allowed


@torch.no_grad()
def decode_prefix(model, prefix, memory, source_allowed):
    _require_eval(model)
    # A generated prefix may contain repeated EOS after an example finishes,
    # but never contains PAD. Finished rows are masked only in returned output.
    valid = torch.ones_like(prefix, dtype=torch.bool)
    _, target_allowed = build_masks(source_allowed[:, 0, 0], valid)
    x = model._embed(prefix, model.target_embedding)
    for layer in model.decoder_layers:
        x, _, _ = layer(x, memory, target_allowed, source_allowed)
    return model.vocabulary_projection(x)


def _attend(attention, q, k, v, allowed=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(attention.head_width)
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))
    return attention.out_proj(attention.merge_heads(torch.softmax(scores, -1) @ v))


class CachedDecoder:
    """Per-request, per-layer self-KV and static cross-KV caches.

    `step` consumes exactly one [B, 1] token column at the next absolute
    position. It returns [B, V] logits. No multi-token prefill is hidden here;
    use successive steps to prefill a longer target prefix.
    """

    @torch.no_grad()
    def __init__(self, model, memory, source_allowed):
        _require_eval(model)
        self.model, self.source_allowed = model, source_allowed
        self.position = 0
        self.self_kv = [None] * len(model.decoder_layers)
        self.cross_kv = []
        for layer in model.decoder_layers:
            attention = layer.cross_attention
            self.cross_kv.append((
                attention.split_heads(attention.k_proj(memory)),
                attention.split_heads(attention.v_proj(memory)),
            ))

    @torch.no_grad()
    def step(self, token):
        _require_eval(self.model)
        if token.ndim != 2 or token.shape[1] != 1:
            raise ValueError("token must have shape [batch, 1]")
        if token.shape[0] != self.source_allowed.shape[0] or (token == PAD).any():
            raise ValueError("batch must match memory and token cannot be PAD")
        if self.position >= self.model.position.pe.shape[1]:
            raise ValueError("sequence exceeds max_length")
        model = self.model
        x = model.target_embedding(token) * math.sqrt(model.d_model)
        x = x + model.position.pe[:, self.position:self.position + 1].to(x.dtype)
        # Dropout is identity in eval mode. Preserve the post-norm order.
        for i, layer in enumerate(model.decoder_layers):
            attention = layer.self_attention
            # snippet:cache_step
            q = attention.split_heads(attention.q_proj(x))
            new_k = attention.split_heads(attention.k_proj(x))
            new_v = attention.split_heads(attention.v_proj(x))
            if self.self_kv[i] is None:
                k, v = new_k, new_v
            else:
                old_k, old_v = self.self_kv[i]
                k = torch.cat((old_k, new_k), dim=2)
                v = torch.cat((old_v, new_v), dim=2)
            self.self_kv[i] = (k, v)
            attended = _attend(attention, q, k, v)
            x = layer.norm1(x + attended)
            # end-snippet:cache_step
            cross = layer.cross_attention
            q = cross.split_heads(cross.q_proj(x))
            k, v = self.cross_kv[i]
            x = layer.norm2(x + _attend(cross, q, k, v, self.source_allowed))
            x = layer.norm3(x + layer.feed_forward(x))
        self.position += 1
        return model.vocabulary_projection(x[:, 0])


@torch.no_grad()
def greedy_decode(model, source, max_new_tokens=8, cached=True):
    _require_eval(model)
    if max_new_tokens < 1 or max_new_tokens > model.position.pe.shape[1]:
        raise ValueError("max_new_tokens must be positive and within max_length")
    memory, source_allowed = encode_source(model, source)
    cache = CachedDecoder(model, memory, source_allowed) if cached else None
    # snippet:greedy_decode
    batch_size = source.shape[0]
    prefix = source.new_full((batch_size, 1), BOS)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=source.device)
    generated = []
    for _ in range(max_new_tokens):
        logits = (cache.step(prefix[:, -1:]) if cached else
                  decode_prefix(model, prefix, memory, source_allowed)[:, -1])
        logits[:, [PAD, BOS]] = float("-inf")
        selected = logits.argmax(dim=-1)
        # Keep finished rows inert. EOS remains a valid cache input.
        selected = torch.where(finished, EOS, selected)
        generated.append(torch.where(finished, PAD, selected))
        finished |= selected == EOS
        prefix = torch.cat((prefix, selected[:, None]), dim=1)
        if finished.all():
            break
    return torch.stack(generated, dim=1)
    # end-snippet:greedy_decode


@torch.no_grad()
def benchmark(model, source, prefix, repeats=15):
    """Median CPU milliseconds for fixed work, excluding data loading.

    Source encoding and target prefill (all but the final prefix token) are
    reported separately. Decode times consume that final supplied token;
    they do not include sampling or detokenization. Rebuild caches outside
    each decode timing so every repetition starts at the same position.
    """
    _require_eval(model)
    if source.device.type != "cpu" or prefix.device.type != "cpu":
        raise ValueError("this benchmark is CPU-only")
    if prefix.shape[1] < 2 or repeats < 1:
        raise ValueError("prefix needs >=2 tokens and repeats must be positive")
    def median_ms(run):
        times = []
        for _ in range(repeats):
            start = perf_counter()
            run()
            times.append(1000 * (perf_counter() - start))
        return statistics.median(times)
    for _ in range(3):
        memory, allowed = encode_source(model, source)
        decode_prefix(model, prefix, memory, allowed)
        cache = CachedDecoder(model, memory, allowed)
        for t in range(prefix.shape[1]):
            cache.step(prefix[:, t:t + 1])
    encoding_ms = median_ms(lambda: encode_source(model, source))
    memory, allowed = encode_source(model, source)
    def prefill():
        cache = CachedDecoder(model, memory, allowed)
        for t in range(prefix.shape[1] - 1):
            cache.step(prefix[:, t:t + 1])
        return cache
    prefill_ms = median_ms(prefill)
    full_ms = median_ms(lambda: decode_prefix(model, prefix, memory, allowed))
    times = []
    for _ in range(repeats):
        cache = prefill()
        start = perf_counter()
        cache.step(prefix[:, -1:])
        times.append(1000 * (perf_counter() - start))
    return {"source_encode_ms": encoding_ms, "cached_target_prefill_ms": prefill_ms,
            "full_prefix_next_token_ms": full_ms,
            "cached_next_token_ms": statistics.median(times),
            "prefix_tokens": prefix.shape[1], "batch_size": source.shape[0],
            "source_columns": source.shape[1],
            "source_valid_tokens": int((source != model.pad_id).sum().item()),
            "repeats": repeats, "threads": torch.get_num_threads()}
