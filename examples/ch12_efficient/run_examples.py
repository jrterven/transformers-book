"""Run the Chapter 12 numerical checks on a CPU."""

import argparse
import json
from pathlib import Path

import numpy as np

from examples.ch04_attention.attention import scaled_dot_product_attention
from examples.ch10_kv.model import causal_mask

from .model import online_update, score_workspace, tiled_attention


def online_example():
    maximum = np.array([-np.inf])
    denominator = np.zeros(1)
    numerator = np.zeros((1, 1))
    states = []
    for scores, values, allowed in (
        ([[0.0, np.log(2)]], [[1.0], [2.0]], None),
        ([[1000.0]], [[999.0]], [[False]]),
        ([[np.log(3)]], [[4.0]], None),
    ):
        maximum, denominator, numerator = online_update(
            maximum, denominator, numerator, scores, values, allowed
        )
        states.append({
            "maximum": float(maximum[0]),
            "denominator": float(denominator[0]),
            "numerator": float(numerator[0, 0]),
        })
    return {"states": states, "output": float(numerator[0, 0] / denominator[0])}


def projected_example():
    rng = np.random.default_rng(12)
    return (
        rng.standard_normal((7, 3)),
        rng.standard_normal((9, 3)),
        rng.standard_normal((9, 2)),
    )


def attention_example():
    q, k, v = projected_example()
    query_start = 2
    allowed = causal_mask(query_start, len(q), len(k))
    results = {}
    for name, mask in (("unmasked", None), ("causal", allowed)):
        dense, _ = scaled_dot_product_attention(q, k, v, mask)
        tiled = tiled_attention(
            q, k, v, query_tile=3, key_tile=4,
            causal=mask is not None, query_start=query_start,
        )
        results[name] = {
            "output_shape": list(tiled.shape),
            "max_absolute_output_difference": float(np.max(np.abs(tiled - dense))),
        }
    return results


def workspace_example():
    full_small, tile_small = score_workspace(7, 9, 3, 4)
    full_large, tile_large = score_workspace(4096, 4096, 64, 128)
    return {
        "small_full_score_elements": full_small,
        "small_score_tile_elements": tile_small,
        "large_full_score_bytes_at_two_bytes_per_element": 2 * full_large,
        "large_score_tile_bytes_at_two_bytes_per_element": 2 * tile_large,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "online_softmax": online_example(),
        "attention": attention_example(),
        "one_score_array": workspace_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
