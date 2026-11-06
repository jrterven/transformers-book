"""Run Chapter 26's fixed candidates and fill-in-the-middle format example."""

import argparse
import json
from pathlib import Path

from .model import (
    Case,
    needs_two,
    pass_at_k,
    positive_length,
    score_candidate,
    serialize_fim,
    strict_overlap,
    touching_counts,
)


CANDIDATES = (touching_counts, strict_overlap, needs_two, positive_length)
SMOKE = (
    Case("interior_overlap", (0, 4), (2, 6), True),
    Case("separated", (0, 2), (3, 5), False),
)
EDGES = (
    Case("touching", (0, 2), (2, 5), False),
    Case("empty", (2, 2), (0, 4), False),
    Case("length_one", (0, 3), (2, 4), True),
    Case("negative_endpoints", (-6, -2), (-4, 1), True),
)


def benchmark_example():
    candidates = []
    for candidate in CANDIDATES:
        smoke = score_candidate(candidate, SMOKE)
        edges = score_candidate(candidate, EDGES)
        candidates.append({
            "name": candidate.__name__,
            "smoke": smoke,
            "edges": edges,
            "complete_passed_cases": smoke["passed_cases"] + edges["passed_cases"],
            "complete_passed_all": smoke["passed_all"] and edges["passed_all"],
        })
    n = len(candidates)
    smoke_correct = sum(row["smoke"]["passed_all"] for row in candidates)
    complete_correct = sum(row["complete_passed_all"] for row in candidates)
    return {
        "candidates": candidates,
        "n": n,
        "smoke_correct": smoke_correct,
        "complete_correct": complete_correct,
        "complete_pass_at_1": pass_at_k(n, complete_correct, 1),
        "complete_pass_at_2": pass_at_k(n, complete_correct, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "benchmark": benchmark_example(),
        "fim": serialize_fim("def double(n):\n    ", "\n\nresult = double(3)\n"),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
