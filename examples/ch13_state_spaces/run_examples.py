"""Check the Chapter 13 worked examples on a CPU."""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import (
    bilinear_discretize,
    causal_convolution,
    fft_causal_convolution,
    hyena_stage,
    ssm_kernel,
    ssm_scan,
)


def ssm_example(u, a_bar, b_bar, c):
    outputs, states = ssm_scan(u, a_bar, b_bar, c)
    kernel = ssm_kernel(a_bar, b_bar, c, len(u))
    direct = causal_convolution(u, kernel)
    fft = fft_causal_convolution(u, kernel)
    return {
        "inputs": np.asarray(u).tolist(),
        "states": states.tolist(),
        "kernel": kernel.tolist(),
        "scan": outputs.tolist(),
        "direct_convolution": direct.tolist(),
        "fft_convolution": fft.tolist(),
        "direct_max_error": float(np.max(np.abs(outputs - direct))),
        "fft_max_error": float(np.max(np.abs(outputs - fft))),
    }


def scalar_example():
    u = (2, 0, 4, 0)
    a_bar, b_bar, c = [[0.5]], [1], [1]
    result = ssm_example(u, a_bar, b_bar, c)
    with_x0, _ = ssm_scan(u, a_bar, b_bar, c, x0=[2])
    initial_response, _ = ssm_scan(np.zeros(len(u)), a_bar, b_bar, c, x0=[2])
    result.update({
        "scan_from_state_2": with_x0.tolist(),
        "initial_state_response": initial_response.tolist(),
    })
    return result


def vector_example():
    return ssm_example(
        (2, 0, 4, 0), [[0.5, 0.25], [0, 0.25]], [1, 2], [1, -1]
    )


def padding_example():
    u = [1, 0, 0, 2]
    kernel = [1, 1, 1]
    # A short FFT computes the circular convolution for comparison.
    short_spectrum = np.fft.rfft(u) * np.fft.rfft(kernel, n=len(u))
    circular = np.fft.irfft(short_spectrum, n=len(u))
    return {
        "inputs": u,
        "kernel": kernel,
        "fft_length": len(u) + len(kernel) - 1,
        "padded_fft": fft_causal_convolution(u, kernel).tolist(),
        "circular_at_input_length": circular.tolist(),
    }


def bilinear_example():
    a_bar, b_bar = bilinear_discretize([[-2]], [1], 0.5)
    return {"delta": 0.5, "a_bar": a_bar.tolist(), "b_bar": b_bar.tolist()}


def hyena_example():
    gated, filtered, outputs = hyena_stage(
        [1, 2, 0, 1], [1, 0, 1, 1], [1, 1, 2, 1], [1, 0.5, 0.25, 0.125]
    )
    return {
        "gated_input": gated.tolist(),
        "filtered": filtered.tolist(),
        "output": outputs.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "scalar_ssm": scalar_example(),
        "vector_ssm": vector_example(),
        "padding": padding_example(),
        "bilinear": bilinear_example(),
        "hyena": hyena_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
