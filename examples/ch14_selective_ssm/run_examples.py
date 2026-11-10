"""Run Chapter 14's selective and structured state space examples."""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import (
    compose_diagonal,
    selective_scan,
    ssd_chunked,
    ssd_dense,
    ssd_recurrent,
    ssd_head_chunked,
    ssd_head_recurrent,
    zoh_coefficients,
)


def selective_example():
    u = np.array([2.0, 9.0, 4.0, 7.0])
    delta = -np.log([0.5, 0.99, 0.5, 0.99])
    a, b = zoh_coefficients(delta)
    _, states = selective_scan(u, a[:, None], b[:, None], np.ones((len(u), 1)))
    combined_a, combined_write = compose_diagonal(
        (a[0], b[0] * u[0]), (a[1], b[1] * u[1])
    )
    tiny_a, tiny_b = zoh_coefficients(1e-20)
    return {
        "inputs": u.tolist(),
        "steps": delta.tolist(),
        "retention": a.tolist(),
        "write_coefficients": b.tolist(),
        "states": states[:, 0].tolist(),
        "first_two_combined": [float(combined_a), float(combined_write)],
        "small_step": {
            "step": 1e-20,
            "write_using_expm1": float(tiny_b),
            "write_using_subtraction": float(1 - tiny_a),
        },
    }


def ssd_inputs():
    u = np.array([2.0, 0.0, 4.0, 0.0])
    a = np.full(len(u), 0.5)
    b = np.array([[1, 0], [0, 1], [1, 1], [1, 0]])
    c = np.array([[1, 0], [1, 1], [0, 1], [1, 1]])
    return u, a, b, c


def ssd_example():
    u, a, b, c = ssd_inputs()
    recurrent, states = ssd_recurrent(u, a, b, c)
    dense, matrix = ssd_dense(u, a, b, c)
    chunked, final_state = ssd_chunked(u, a, b, c, chunk_size=2)
    within_second, _ = ssd_dense(u[2:], a[2:], b[2:], c[2:])
    return {
        "recurrent": recurrent.tolist(),
        "states": states.tolist(),
        "dense": dense.tolist(),
        "matrix": matrix.tolist(),
        "chunked": chunked.tolist(),
        "chunked_final_state": final_state.tolist(),
        "second_chunk_within": within_second.tolist(),
        "second_chunk_from_incoming_state": (chunked[2:] - within_second).tolist(),
    }


def warm_ssd_example():
    _, _, b, c = ssd_inputs()
    u = [2, 0, 4, 0, 1]
    a = [0.5, 0.25, 0.5, 0.0, 0.75]
    b = np.vstack((b, [2, -1]))
    c = np.vstack((c, [1, -1]))
    x0 = [2, -1]
    recurrent, states = ssd_recurrent(u, a, b, c, x0)
    dense, _ = ssd_dense(u, a, b, c, x0)
    chunked, final_state = ssd_chunked(u, a, b, c, 2, x0)
    return {
        "retention": a,
        "initial_state": x0,
        "chunk_lengths": [2, 2, 1],
        "recurrent": recurrent.tolist(),
        "dense": dense.tolist(),
        "chunked": chunked.tolist(),
        "recurrent_final_state": states[-1].tolist(),
        "chunked_final_state": final_state.tolist(),
    }


def head_example():
    first_feature, a, b, c = ssd_inputs()
    u = np.column_stack([first_feature, [-1, 2, 0, 1]])
    recurrent, states = ssd_head_recurrent(u, a, b, c)
    chunked, last = ssd_head_chunked(u, a, b, c, chunk_size=3)
    _, matrix = ssd_dense(first_feature, a, b, c)
    np.testing.assert_allclose(chunked, matrix @ u, atol=1e-14, rtol=0)
    np.testing.assert_allclose(recurrent, chunked, atol=1e-14, rtol=0)
    np.testing.assert_allclose(last, states[-1], atol=1e-14, rtol=0)
    return {
        "inputs": u.tolist(), "outputs": recurrent.tolist(),
        "state_shape": list(last.shape), "final_state": last.tolist(),
        "chunk_lengths": [3, 1], "dense_max_error": float(np.max(np.abs(chunked - matrix @ u))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "selective": selective_example(),
        "ssd": ssd_example(),
        "ssd_with_initial_state": warm_ssd_example(),
        "two_feature_head": head_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
