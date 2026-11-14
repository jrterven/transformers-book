"""Run Chapter 26's fixed diffusion examples without model downloads."""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import add_noise, ddim_step, guided_noise


def denoising_example():
    clean = np.array([[1.0, 0], [0, 1]])
    noise = np.array([[0.0, 1], [-1, 0]])
    start, middle, end = 0.36, 0.64, 1.0
    noisy = add_noise(clean, noise, start)

    def two_steps(prediction):
        estimated_clean, next_noisy = ddim_step(noisy, prediction, start, middle)
        _, final = ddim_step(next_noisy, prediction, middle, end)
        return {
            "predicted_noise": prediction.tolist(),
            "estimated_clean_from_first_step": estimated_clean.tolist(),
            "next_noisy": next_noisy.tolist(),
            "final": final.tolist(),
            "final_max_absolute_error": float(np.max(np.abs(final - clean))),
        }

    perturbed = noise.copy()
    perturbed[0, 0] += 0.25
    return {
        "clean": clean.tolist(),
        "noise": noise.tolist(),
        "alpha_bar_schedule": [start, middle, end],
        "noisy": noisy.tolist(),
        "exact_prediction": two_steps(noise),
        "perturbed_prediction": two_steps(perturbed),
    }


def guidance_example():
    unconditional, conditional, scale = [0.2, -0.2], [0.6, 0.2], 2
    return {
        "unconditional": unconditional,
        "conditional": conditional,
        "scale": scale,
        "guided": guided_noise(unconditional, conditional, scale).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {"denoising": denoising_example(), "guidance": guidance_example()}
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
