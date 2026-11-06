"""Run Chapter 17's deterministic evaluation examples without downloads."""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import (
    binary_metrics,
    paired_accuracy_bootstrap,
    reliability_bins,
    split_by_group,
)


def classification_data():
    # Every prediction and group refers to the same row in this fixture.
    row_ids = np.arange(1, 13)
    groups = np.array(["a", "a", "b", "b", "b", "c", "d", "d", "e", "e", "e", "f"])
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    pred_a = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    pred_b = np.array([1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    return row_ids, groups, y_true, pred_a, pred_b


def group_example():
    row_ids, groups, _, _, _ = classification_data()
    parts = split_by_group(groups, (3, 1, 2), seed=17)
    return {
        name: {
            "row_ids": row_ids[part].tolist(),
            "groups": np.unique(groups[part]).tolist(),
        }
        for name, part in zip(("train", "validation", "test"), parts)
    }


def metric_example():
    _, _, y_true, pred_a, pred_b = classification_data()
    return {"model_a": binary_metrics(y_true, pred_a),
            "model_b": binary_metrics(y_true, pred_b)}


def calibration_example():
    scores = [0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9]
    truth = [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
    edges = [0, 1 / 3, 2 / 3, 1]
    counts, means, rates, brier = reliability_bins(truth, scores, edges)
    return {
        "edges": edges,
        "counts": counts.tolist(),
        "mean_scores": [float(x) if np.isfinite(x) else None for x in means],
        "positive_rates": [float(x) if np.isfinite(x) else None for x in rates],
        "brier_score": brier,
    }


def bootstrap_example():
    row_ids, groups, y_true, pred_a, pred_b = classification_data()
    row_point, row_interval = paired_accuracy_bootstrap(y_true, pred_a, pred_b)
    group_point, group_interval = paired_accuracy_bootstrap(y_true, pred_a, pred_b, groups=groups)
    return {
        "row_ids": row_ids.tolist(),
        "group_ids": groups.tolist(),
        "accuracy_gap_by_row": (
            (pred_b == y_true).astype(int) - (pred_a == y_true).astype(int)
        ).tolist(),
        "rows": {"point": row_point, "interval": list(row_interval)},
        "groups": {"point": group_point, "interval": list(group_interval)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "group_split": group_example(),
        "binary_metrics": metric_example(),
        "calibration": calibration_example(),
        "paired_accuracy_bootstrap": bootstrap_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
