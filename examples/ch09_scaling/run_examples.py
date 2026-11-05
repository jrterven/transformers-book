"""Compare dense allocations, then trace tokens through a four-expert bank."""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import BILLION, allocation_sweep, expert_branch, route_top1, training_flops


BUDGET = 2.4e22
MODEL_SIZES_B = (5, 10, 20, 40, 80)


def budget_example():
    rows = allocation_sweep(BUDGET, MODEL_SIZES_B)
    return {
        "budget_flops": BUDGET,
        "candidates": [
            {"parameters_b": n, "tokens_b": d,
             "estimated_flops": training_flops(n * BILLION, d * BILLION),
             "toy_loss": loss}
            for n, d, loss in rows
        ],
        "best_candidate_parameters_b": min(rows, key=lambda row: row[2])[0],
    }


def teaching_weights():
    router_weight = np.log([
        [0.1, 0.6, 0.2, 0.1],
        [0.6, 0.2, 0.1, 0.1],
        [0.1, 0.2, 0.6, 0.1],
        [0.1, 0.2, 0.1, 0.6],
    ])
    # Each expert copies one input coordinate into its first hidden unit.
    w_up = np.zeros((4, 4, 8))
    w_up[np.arange(4), [1, 0, 2, 3], 0] = 1
    w_down = np.zeros((4, 8, 4))
    w_down[:, 0] = [[1, 0, 0, 0], [2, -1, 0, 3], [0, 0, 3, 0], [0, 0, 0, 4]]
    return router_weight, w_up, w_down


def token_example():
    x = np.array([[1., 0., 0., 0.]])
    router_weight, w_up, w_down = teaching_weights()
    probabilities, selected, gates, accepted = route_top1(x @ router_weight, 1)
    branch = expert_branch(x, selected, gates, accepted, w_up, w_down)
    expert = selected[0]
    return {
        "input": x[0].tolist(),
        "probabilities": probabilities[0].tolist(),
        "expert": int(expert) + 1,
        "gate": float(gates[0]),
        "expert_output": (np.maximum(x @ w_up[expert], 0) @ w_down[expert])[0].tolist(),
        "branch": branch[0].tolist(),
        "after_residual": (x + branch)[0].tolist(),
        "stored_expert_weights": w_up.size + w_down.size,
        "selected_expert_weights": w_up[expert].size + w_down[expert].size,
        "router_weights": router_weight.size,
    }


def capacity_example(capacity=3):
    x = np.eye(4)[[1, 0, 0, 0, 0, 0, 2, 3]]
    router_weight, w_up, w_down = teaching_weights()
    probabilities, selected, gates, accepted = route_top1(x @ router_weight, capacity)
    branch = expert_branch(x, selected, gates, accepted, w_up, w_down)
    experts = probabilities.shape[1]
    requested = np.bincount(selected, minlength=experts)
    processed = np.bincount(selected[accepted], minlength=experts)
    fraction = requested / len(x)
    mean_probability = probabilities.mean(axis=0)
    return {
        "capacity": capacity,
        "requests": (selected + 1).tolist(),
        "requested_per_expert": requested.tolist(),
        "processed_per_expert": processed.tolist(),
        "overflow_rows": (np.flatnonzero(~accepted) + 1).tolist(),
        "requested_fraction": fraction.tolist(),
        "mean_probability": mean_probability.tolist(),
        "raw_balance_term": float(experts * (fraction @ mean_probability)),
        "input": x.tolist(),
        "branch": branch.tolist(),
        "after_residual": (x + branch).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "dense_budget": budget_example(),
        "one_token": token_example(),
        "capacity_batch": capacity_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
