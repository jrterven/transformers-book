# Chapter 9. Scaling Data, Compute, and Expert Capacity

Run from the repository root with the NumPy requirements.

```sh
python -m pip install -r requirements.txt
python -m examples.ch09_scaling.run_examples --output output/ch09.json
python -m unittest discover -s tests -p test_scaling.py -v
```

The examples use arithmetic and fixed weights on a CPU. They report a
synthetic loss and expert routes that can be checked by hand.

## Spend the same budget five ways

`training_flops` applies the dense-model approximation `C = 6 * N * D`.
It takes raw parameter and token counts. `tokens_at_budget` solves for
`D` as a continuous quantity. A training run would round to whole tokens
and account for model-specific work and system overhead separately.

`allocation_sweep` takes candidate model sizes in billions. It converts
each size to raw counts for the FLOP calculation, then converts the
resulting token count back to billions for the illustrative loss
`1 + 0.4 / sqrt(N_B) + 2 / sqrt(D_B)`. Under the example budget of
`2.4e22` FLOPs, the candidate with 10 billion parameters and 400 billion
tokens has the lowest synthetic loss among the five tested allocations.

## Follow one token and then a batch

The router projects rows `x` of shape `[tokens, width]` using weights
`W_R` of shape `[width, experts]`. In `route_top1`, the softmax acts
over experts independently for each token. Each row keeps its highest
probability as a gate. IDs in the arrays start at zero. The printed
expert labels and token-row numbers start at one.

For each expert, the first requests in input order fill its capacity.
`expert_branch` gathers those rows, applies a two-matrix ReLU network,
multiplies by the selected gates, and writes the updates back into their
original rows. The bank has weight shapes `[experts, width, hidden]` and
`[experts, hidden, width]`. The caller adds the branch to `x` for the
residual update.

In the first example, expert 2 produces `(2, -1, 0, 3)`, and its gate
`0.6` scales this to `(1.2, -0.6, 0, 1.8)`. For this bank, 256 expert
weights are stored, a single routed token uses 64 expert weights, and
the router has 16 weights.

In the batch example, experts receive requests `(1, 5, 1, 1)` and each
can process three rows. Rows 5 and 6 have zero branch updates, so the
residual preserves their inputs. Change `capacity_example(3)` to
`capacity_example(2)` or `capacity_example(5)` to compare the number
of processed rows. The output also gives the expert assignment fractions
and mean probabilities before capacity filtering, and their raw balance
term before any training coefficient is applied.
