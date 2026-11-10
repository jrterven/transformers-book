"""Run the Chapter 15 delta update and streaming stack examples."""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import (delta_chunked, delta_sequence, delta_step, history_counts,
                    initial_stores, stack_step)


HYBRID = ("delta", "attention", "delta")


def delta_example():
    state = np.array([[2.0, 0.0], [1.0, 3.0]])
    key = np.array([1.0, 0.0])
    value = np.array([4.0, 1.0])
    alpha = beta = 0.5
    output, updated = delta_step(state, key, key, value, alpha, beta)
    orthogonal, _ = delta_step(state, [0, 1], key, value, alpha, beta)
    decayed = alpha * state
    prediction = decayed @ key
    return {
        "decayed": decayed.tolist(),
        "prediction": prediction.tolist(),
        "error": (value - prediction).tolist(),
        "updated_state": updated.tolist(),
        "key_read": output.tolist(),
        "orthogonal_read": orthogonal.tolist(),
    }


def negative_carry_example():
    state = np.eye(2)
    key = np.array([1.0, 0.0])
    output, updated = delta_step(state, key, key, [0, 0], alpha=0.8, beta=1.5)
    return {
        "updated_state": updated.tolist(),
        "key_read": output.tolist(),
        "orthogonal_read": (updated @ [0, 1]).tolist(),
    }


def toy_tokens():
    return np.array([
        [1, 0], [0, 1], [1, 1], [2, -1],
        [-1, 2], [0.5, 1], [-1, -1], [1, 2],
    ])


def chunk_example():
    keys = np.array([[1., 0.], [0.6, 0.8], [0., 1.]])
    values = np.array([[2.], [5.], [1.]])
    alpha = np.array([1., 0.5, 1.])
    beta = np.array([1., 1., 0.5])
    sequential, state = delta_sequence(keys, keys, values, alpha, beta)
    chunked, boundary = delta_chunked(keys, keys, values, alpha, beta, 2)
    np.testing.assert_allclose(sequential, chunked, rtol=0, atol=1e-14)
    np.testing.assert_allclose(state, boundary, rtol=0, atol=1e-14)
    return {"outputs": sequential.tolist(), "final_state": state.tolist(),
            "chunked_outputs": chunked.tolist(), "chunk_size": 2}


def schedule_example(tokens, schedule):
    schedule = tuple(schedule)
    stores = initial_stores(schedule)
    counts = [list(history_counts(stores))]
    fourth_trace = last_output = None
    for position, token in enumerate(tokens, start=1):
        trace, stores = stack_step(token, schedule, stores)
        if position == 4:
            fourth_trace = [row.tolist() for row in trace]
        last_output = trace[-1].tolist()
        counts.append(list(history_counts(stores)))
    return {
        "schedule": list(schedule),
        "history_counts_by_prefix": counts,
        "total_history_by_prefix": [sum(layer_counts) for layer_counts in counts],
        "fourth_token_representations": fourth_trace,
        "last_output": last_output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    tokens = toy_tokens()
    result = {
        "delta_update": delta_example(),
        "unit_key_negative_carry": negative_carry_example(),
        "chunked_delta": chunk_example(),
        "schedules": {
            "all_delta": schedule_example(tokens, ("delta",) * 3),
            "hybrid": schedule_example(tokens, HYBRID),
            "all_attention": schedule_example(tokens, ("attention",) * 3),
        },
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
