"""Run the Chapter 11 connectivity, selection, and compression examples."""

import argparse
import json
from pathlib import Path

import numpy as np

from examples.ch04_attention.attention import scaled_dot_product_attention

from .model import block_summaries, local_mask, selected_attention


def connectivity_example():
    length = 8
    local = local_mask(length, 1)
    with_global = local_mask(length, 1, global_positions=(0,))
    return {
        "length": length,
        "dense_connections": length**2,
        "local_connections": int(local.sum()),
        "local_and_global_connections": int(with_global.sum()),
        "local_keys_at_position_3": np.flatnonzero(local[3]).tolist(),
    }


def selection_example():
    query = np.array([1.0, 0.0])
    keys = np.zeros((8, 2))
    values = np.zeros((8, 2))
    keys[[1, 6]] = [[np.sqrt(2) * np.log(3), 0], [0, 1]]
    values[[1, 6]] = [[1, 0], [0, 2]]
    index_scores = np.array([0.2, 2.4, 0.1, 0.7, 0.3, 0.5, 1.8, 0.4])
    output, selected, weights = selected_attention(
        query, keys, values, index_scores, 2
    )

    # Give the future position a high score and then exclude it by time.
    future_scores = index_scores.copy()
    future_scores[-1] = 99
    allowed = np.arange(len(keys)) <= 6
    _, causal_selected, _ = selected_attention(
        query, keys, values, future_scores, 2, allowed
    )
    all_output, _, _ = selected_attention(
        query, keys, values, future_scores, len(keys), allowed
    )
    dense_output, _ = scaled_dot_product_attention(
        query[None, :], keys, values, allowed
    )
    return {
        "index_scores": index_scores.tolist(),
        "selected": selected.tolist(),
        "main_scores": (keys[selected] @ query / np.sqrt(len(query))).tolist(),
        "main_weights": weights.tolist(),
        "output": output.tolist(),
        "causal_selected_with_large_future_score": causal_selected.tolist(),
        "all_allowed_output": all_output.tolist(),
        "masked_dense_output": dense_output[0].tolist(),
        "all_allowed_max_output_difference": float(np.max(
            np.abs(all_output - dense_output[0])
        )),
    }


def compression_example():
    features = np.array([[1.0], [3.0], [2.0], [6.0]])
    gates = np.log([[0.1], [0.2], [0.3], [0.4]])
    one_summary, _ = block_summaries(features, gates, 4)

    history = np.arange(1.0, 17.0)[:, None]
    uniform_gates = np.zeros_like(history)
    _, overlap_ends = block_summaries(history, uniform_gates, 2, overlap=True)
    _, disjoint_ends = block_summaries(history, uniform_gates, 4)
    return {
        "gated_scalar_summary": float(one_summary[0, 0]),
        "stride_2_overlap_ends": overlap_ends.tolist(),
        "stride_4_disjoint_ends": disjoint_ends.tolist(),
        "raw_recent_positions": list(range(12, 16)),
    }


def mixed_history_example():
    features = np.array([[1.0], [3.0], [2.0], [6.0], [4.0], [5.0], [8.0], [100.0]])
    summaries, ends = block_summaries(features, np.zeros_like(features), 2)
    query_position = 6
    visible = np.flatnonzero(ends <= query_position)
    index_scores = np.array([1.8, 0.4, 0.3, 9.0])
    selected = visible[np.argmax(index_scores[visible])]
    recent = np.arange(query_position - 1, query_position + 1)

    # One softmax across both types of memory entry.
    memory = np.concatenate((summaries[[selected]], features[recent]))
    output, weights = scaled_dot_product_attention([[0.0]], memory, memory)
    return {
        "query_position": query_position,
        "summary_ends": ends.tolist(),
        "visible_summary_ends": ends[visible].tolist(),
        "selected_summary": int(selected),
        "raw_recent_positions": recent.tolist(),
        "gathered_values": memory[:, 0].tolist(),
        "main_weights": weights[0].tolist(),
        "output": float(output[0, 0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "connectivity": connectivity_example(),
        "selection": selection_example(),
        "compression": compression_example(),
        "mixed_history": mixed_history_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
