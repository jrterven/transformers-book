"""Run Chapter 28's row encoding and fixed probability examples."""

import argparse
import json
import math
from pathlib import Path

from .model import brier_score, context_probability, encode_rows, fit_row_encoder


TRAIN_NUMERIC = (2, 4, None, 8)
TRAIN_CATEGORIES = ("web", "shop", None, "web")
HELD_OUT_NUMERIC = (None, 6, 100)
HELD_OUT_CATEGORIES = ("kiosk", "shop", None)


def row_example():
    median, levels = fit_row_encoder(TRAIN_NUMERIC, TRAIN_CATEGORIES)
    train = encode_rows(TRAIN_NUMERIC, TRAIN_CATEGORIES, median, levels)
    held_out = encode_rows(HELD_OUT_NUMERIC, HELD_OUT_CATEGORIES, median, levels)
    return {
        "median": median,
        "levels": levels,
        "columns": ("numeric_filled", "numeric_missing", *levels,
                    "category_missing", "category_unknown"),
        "train_features": train.tolist(),
        "held_out_features": held_out.tolist(),
    }


def probability_example():
    logits, labels = [0, math.log(2), math.log(3)], [0, 1, 1]
    probability, weights = context_probability(logits, labels)
    probabilities = [0.1, 0.6, 0.8]
    return {
        "context_labels": labels,
        "context_weights": weights.tolist(),
        "context_probability": probability,
        "example_probabilities": probabilities,
        "brier_score": brier_score(labels, probabilities),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {"encoding": row_example(), "probabilities": probability_example()}
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
