"""Count-based policies and a small deterministic grid environment."""

from collections import Counter, defaultdict
from numbers import Integral, Real

import numpy as np


ROWS, COLS = 2, 3
START, GOAL = (0, 0), (0, 2)
ORIGINAL_TRAP = (0, 1)
MOVES = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}
ACTIONS = tuple(MOVES)  # Also the order used to break ties.


def _in_grid(row, col):
    return 0 <= row < ROWS and 0 <= col < COLS


def _grid_cell(cell):
    try:
        row, col = cell
    except (TypeError, ValueError) as error:
        raise ValueError("a cell needs two integer coordinates") from error
    if any(not isinstance(index, Integral) or isinstance(index, bool) for index in (row, col)):
        raise ValueError("a cell needs two integer coordinates")
    row, col = int(row), int(col)
    if not _in_grid(row, col):
        raise ValueError("cell is outside the grid")
    return row, col


def _grid_step(state, action, trap):
    row, col = state
    dr, dc = MOVES[action]
    destination = (row + dr, col + dc)
    if not _in_grid(*destination):
        destination = state
    if destination == GOAL:
        return destination, 1
    if destination == trap:
        return destination, -1
    return destination, 0


# snippet:returns_to_go
def returns_to_go(rewards):
    """Include the current reward in each undiscounted suffix sum."""
    rewards = np.asarray(rewards)
    if rewards.ndim != 1 or rewards.dtype.kind not in "iuf":
        raise ValueError("rewards must be a one-dimensional real sequence")
    with np.errstate(over="ignore", invalid="ignore"):
        targets = np.cumsum(rewards[::-1], dtype=np.float64)[::-1]
    if not np.all(np.isfinite(targets)):
        raise ValueError("rewards and returns must be finite")
    return targets
# end-snippet:returns_to_go


# snippet:fit_count_policies
def fit_count_policies(trajectories):
    """Count actions from trajectories of (state, action, reward) rows."""
    by_state = defaultdict(Counter)
    by_return = defaultdict(Counter)
    for trajectory in trajectories:
        trajectory = tuple(trajectory)
        targets = returns_to_go([reward for _, _, reward in trajectory])
        for (state, action, _), target in zip(trajectory, targets):
            state = _grid_cell(state)
            if not isinstance(action, str) or action not in MOVES:
                raise ValueError("unknown action")
            by_state[state][action] += 1
            by_return[state, float(target)][action] += 1

    state_policy = {key: max(ACTIONS, key=counts.__getitem__)
                    for key, counts in by_state.items()}
    return_policy = {key: max(ACTIONS, key=counts.__getitem__)
                     for key, counts in by_return.items()}
    return state_policy, return_policy
# end-snippet:fit_count_policies


def rollout(policy, *, trap=ORIGINAL_TRAP, target_return=None, max_steps=8):
    """Start at START and choose each action from the latest observation."""
    trap = _grid_cell(trap)
    if trap in (START, GOAL):
        raise ValueError("trap must differ from the start and goal")
    if (not isinstance(max_steps, Integral) or isinstance(max_steps, bool)
            or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    if target_return is not None:
        if (not isinstance(target_return, Real) or isinstance(target_return, bool)
                or not np.isfinite(float(target_return))):
            raise ValueError("target_return must be a finite real number")
        target_return = float(target_return)

    # snippet:rollout
    state = START
    states, actions, rewards = [state], [], []
    remaining_targets = None if target_return is None else [target_return]
    stop_reason = "step_limit"
    for _ in range(max_steps):
        key = state if target_return is None else (state, target_return)
        action = policy.get(key)
        if action is None:
            stop_reason = "unsupported"
            break
        if not isinstance(action, str) or action not in MOVES:
            raise ValueError("unknown action")

        state, reward = _grid_step(state, action, trap)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        if remaining_targets is not None:
            target_return -= reward
            remaining_targets.append(target_return)
        if state in (GOAL, trap):
            stop_reason = "goal" if state == GOAL else "trap"
            break
    # end-snippet:rollout

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "remaining_targets": remaining_targets,
        "total_reward": sum(rewards),
        "stop_reason": stop_reason,
    }
