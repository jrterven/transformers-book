# Chapter 30. Decisions and Embodied Action

Run the fixed examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch30_decisions.run_examples --output output/ch30.json
python -m unittest discover -s tests -p test_decisions.py -v
```

`returns_to_go` adds each reward to all the rewards that follow
it in the same trajectory. The rewards `[1, -2, 3]` give the
targets `[2, 1, 3]`. An empty sequence gives an empty result.

The four fixed training trajectories in `run_examples.py` record
`(state, action, reward)` at each step. Grid cells use zero-based
`(row, column)` coordinates. The start is `(0, 0)`, the goal is
`(0, 2)`, and the original trap is `(0, 1)`. Reaching the goal
gives `+1`, reaching the trap gives `-1`, and all other moves give
zero. A move into a wall leaves the agent in place.

`fit_count_policies` returns two dictionaries. The first chooses
the most frequent action observed at a state. The second chooses
the most frequent action observed at a state with a given return
to go. Ties use `U, R, D, L` in that order. The conditioned table
looks up an exact target value, which suits the grid's integer
rewards.

`rollout` takes one of the fitted dictionaries and asks it for an
action at each newly observed state. For the conditioned table,
pass `target_return`. After each move, the trace subtracts its
reward from the remaining target. The trace includes the initial
state and, when supplied, the initial target. A missing table entry
stops the rollout with `unsupported` before another action.

Three of the recorded trajectories take `R` into the trap. The
state-only table follows this majority. At target `+1`, the
conditioned table follows the goal route `D, R, R, U` on the
original grid. The same table follows `D, R` into the trap when it
moves to `(1, 1)`. Both tables are simple counts over the recorded
actions. They illustrate how a desired return changes the lookup
and how a changed environment can change the outcome.

## What the examples establish

The reward index refers to the action that earns it. Recorded returns describe
completed episodes; requested returns condition a policy. Neither a large request
nor a valid action token guarantees a successful outcome.

The shifted trap is omitted from the policy inputs. Equal position/return keys
therefore produce equal first actions, despite different consequences.
An unsupported target is distinct from a terminal trap or a time limit.
Target 0 is absent from the log but attainable by a wall policy at the step cap;
target +2 is impossible under this grid's terminal rewards.

The notebook adds a discounted-return extension without changing the canonical
undiscounted helper. It also checks tied demonstrations and separates an
eight-action prediction horizon from the number executed before refreshing.

See the [11 exercises](EXERCISES.md) and
[complete solutions](../../solutions/ch30_decisions/README.md).
