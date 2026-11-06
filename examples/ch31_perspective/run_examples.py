"""Run Chapter 31's memory comparison and causal pair counts."""

import argparse
import json
from pathlib import Path

from .model import causal_pair_counts, compare_memory, make_history


CODES = (731, 954)
ROUTES = tuple("ABCDEFGH")
WINDOW = 3


def memory_examples():
    examples = []
    for code in CODES:
        events = make_history(code, ROUTES[:3])
        examples.append({"events": events, "answers": compare_memory(events, WINDOW)})
    return examples


def accuracy_sweep():
    rows = []
    for updates in (1, 2, 4, 8):
        routes = ROUTES[:updates]
        examples = [compare_memory(make_history(code, routes), WINDOW) for code in CODES]
        methods = {}
        for name in examples[0]:
            answers = [example[name] for example in examples]
            methods[name] = {
                "first_code_accuracy": sum(answer["first_code"] == code
                                           for answer, code in zip(answers, CODES)) / len(CODES),
                "current_route_accuracy": sum(answer["current_route"] == routes[-1]
                                               for answer in answers) / len(CODES),
                "retained_units": answers[0]["retained_units"],
                "unit": answers[0]["unit"],
            }
        rows.append({"route_updates": updates, "cases": len(CODES), "methods": methods})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    pair_counts = []
    for length in (8, 16):
        full, local = causal_pair_counts(length, WINDOW)
        pair_counts.append({"length": length, "full": full, "local": local})
    result = {
        "window": WINDOW,
        "worked_histories": memory_examples(),
        "accuracy_sweep": accuracy_sweep(),
        "causal_pair_counts": pair_counts,
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
