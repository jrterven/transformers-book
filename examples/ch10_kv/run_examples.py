"""Run the Chapter 10 cache and attention examples on a CPU."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .latent import latent_demo
from .model import attention_step, cache_bytes, causal_mask, grouped_attention


def hand_example():
    keys = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    values = np.array([[[[2.0, 0.0], [0.0, 1.0], [2.0, 2.0]]]])
    query = np.array([[[[1.0, 0.0]]]])
    old_cache = (keys[:, :, :2], values[:, :, :2])
    output, weights, _ = attention_step(
        query, keys[:, :, 2:], values[:, :, 2:], old_cache
    )
    scores = (query @ keys.swapaxes(-1, -2)) / math.sqrt(2)
    return {
        "tokens": ["the", "red", "car"],
        "scores": scores[0, 0, 0].tolist(),
        "weights": weights[0, 0, 0].tolist(),
        "output": output[0, 0, 0].tolist(),
    }


def projected_example():
    rng = np.random.default_rng(10)
    return tuple(rng.standard_normal((1, 4, 6, 2)) for _ in range(3))


def run_chunks(q, k, v, chunk_sizes):
    """Process all projected rows in the given sequence of chunks."""
    if sum(chunk_sizes) != q.shape[-2] or not all(n > 0 for n in chunk_sizes):
        raise ValueError("positive chunk sizes must cover every query")
    cache = None
    outputs = []
    start = 0
    for size in chunk_sizes:
        rows = slice(start, start + size)
        output, _, cache = attention_step(
            q[:, :, rows], k[:, :, rows], v[:, :, rows], cache
        )
        outputs.append(output)
        start += size
    return np.concatenate(outputs, axis=-2), cache


def cache_example():
    q, keys, values = projected_example()
    length = q.shape[-2]
    prefill_then_tokens = (2, 1, 1, 1, 1)
    multi_token_chunks = (2, 1, 3)
    results = []
    for kv_heads in (4, 2, 1):
        k, v = keys[:, :kv_heads], values[:, :kv_heads]
        full, _ = grouped_attention(q, k, v, causal_mask(0, length, length))
        single, _ = run_chunks(q, k, v, prefill_then_tokens)
        chunked, (cached_k, cached_v) = run_chunks(q, k, v, multi_token_chunks)
        results.append({
            "query_heads": q.shape[1],
            "kv_heads": kv_heads,
            "cache_shape_each": list(cached_k.shape),
            "output_shape": list(full.shape),
            "cached_scalars": cached_k.size + cached_v.size,
            "cache_array_bytes": cached_k.nbytes + cached_v.nbytes,
            "cache_bytes_at_two_bytes_per_scalar": cache_bytes(
                q.shape[0], 1, length, kv_heads, k.shape[-1], v.shape[-1], 2
            ),
            "single_append_max_error": float(np.max(np.abs(full - single))),
            "chunk_append_max_error": float(np.max(np.abs(full - chunked))),
        })
    return {
        "prefill_then_tokens": list(prefill_then_tokens),
        "multi_token_chunks": list(multi_token_chunks),
        "dtype": q.dtype.name,
        "attention_cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save JSON output to this path")
    args = parser.parse_args()
    result = {
        "hand_example": hand_example(),
        "cache_example": cache_example(),
        "latent_example": latent_demo(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
