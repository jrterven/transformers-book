"""Run the fixed return and grid examples from Chapter 31."""

import argparse
import json
from pathlib import Path

from .model import GOAL, ORIGINAL_TRAP, START, fit_count_policies, returns_to_go, rollout


TRAP_TRAJECTORY = (((0, 0), "R", -1),)
GOAL_TRAJECTORY = (
    ((0, 0), "D", 0),
    ((1, 0), "R", 0),
    ((1, 1), "R", 0),
    ((1, 2), "U", 1),
)
TRAJECTORIES = (TRAP_TRAJECTORY,) * 3 + (GOAL_TRAJECTORY,)
SHIFTED_TRAP = (1, 1)


def grid_example():
    state_policy, return_policy = fit_count_policies(TRAJECTORIES)
    return {
        "start": START,
        "goal": GOAL,
        "original_trap": ORIGINAL_TRAP,
        "shifted_trap": SHIFTED_TRAP,
        "training_trajectories": TRAJECTORIES,
        "state_only": rollout(state_policy),
        "return_conditioned": rollout(return_policy, target_return=1),
        "shifted_return_conditioned": rollout(return_policy, trap=SHIFTED_TRAP, target_return=1),
        "unseen_return": rollout(return_policy, target_return=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    rewards = [1, -2, 3]
    result = {
        "returns": {"rewards": rewards, "returns_to_go": returns_to_go(rewards).tolist()},
        "grid": grid_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
