"""Run Chapter 23's offline matching and evaluation examples."""

import argparse
import json
from pathlib import Path

from .model import all_points_ap, box_iou, mask_iou, match_queries, precision_recall


def overlap_example():
    box = box_iou([0, 0, 2, 2], [1, 0, 3, 2])
    mask = mask_iou(
        [[True, True, False], [False, True, False]],
        [[True, False, False], [False, True, True]],
    )
    return {"box_iou": box, "mask_iou": mask}


def assignment_example():
    cost = [[1, 1], [2, 5], [4, 3]]
    pairs, unmatched, total = match_queries(cost)
    return {
        "cost": cost,
        "pairs": [[f"Q{q + 1}", "AB"[target]] for q, target in pairs],
        "unmatched": [f"Q{q + 1}" for q in unmatched],
        "total_cost": total,
    }


def ap_example():
    ground_truth = [[0, 0, 2, 2], [4, 0, 6, 2]]
    predictions = [ground_truth[0], [0.5, 0, 2.5, 2], ground_truth[1]]
    scores = [0.95, 0.85, 0.70]
    order, true_positive, precision, recall = precision_recall(
        ground_truth, predictions, scores, threshold=0.5
    )
    return {
        "scores": scores,
        "duplicate_iou_with_a": box_iou(predictions[1], ground_truth[0]),
        "ranked_indices": order.tolist(),
        "true_positive": true_positive.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "all_points_ap": all_points_ap(precision, recall),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "overlap": overlap_example(),
        "matching": assignment_example(),
        "ranked_detection": ap_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
