# Chapter 15. Delta Rules and Hybrid Stacks

Run from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch15_hybrids.run_examples --output output/ch15.json
python -m unittest discover -s tests -p test_hybrids.py -v
```

## Predict and correct a stored value

`delta_step` takes a state with shape `[value width, key width]`, one
query and key, one value, and the scalar gates `alpha` and `beta`.
It decays the state, reads a prediction with the key, writes an outer
product of the error and key, and reads with the query. It returns the
output and a new state. The teaching update accepts `alpha` in
`[0, 1]` and `beta` in `[0, 2]`.

The function uses the supplied query and key vectors directly. The
directional carry calculation in the book assumes a unit key. The
program checks this with an identity state, zero value, `alpha=0.8`
and `beta=1.5`. The key direction carries with coefficient `-0.4`
and the orthogonal direction carries with coefficient `0.8`.

## Trace the same token through three layers

Use `initial_stores` with a schedule such as
`("delta", "attention", "delta")` to allocate separate stores. A
recurrent layer starts with a zero `[2, 2]` state. An attention layer
starts with separate empty key and value arrays of width two.

`stack_step` takes one width-two token representation. At each
layer, the query and key projections are the identity, while the
value projection swaps the coordinates. The delta branch explicitly
normalizes its query and key directions, leaving a zero projection
at zero, and uses fixed gates. The full-attention branch appends its
key and value before reading the available history, including the
current position. The function returns the input and each layer
output in order, alongside the new stores.

`history_counts` counts the actual array elements per layer. At a
prefix length of `n`, the three-layer schedules retain `12` scalars
for all-delta, `4n + 8` for the hybrid, and `12n` for all-attention.
The comparison counts persistent mixer history for the toy schedules.
Model parameters, output traces, and temporary workspaces are separate.
The fixed projections and gates let you inspect the history
calculation without training a model.

## Exercises and complete solutions

The [ten exercises](EXERCISES.md) progress from hand calculations to numerical
checks and experiment design. All are solved in the [notebook and both PDFs](../../solutions/ch15_hybrids/README.md).

The exact unit-key overwrite needs both unit key norm and a complete correction.
With decay disabled, repeating the same complete write leaves the updated
state unchanged. General keys can overlap: a correction changes another
read in proportion to the key-query inner product. The low-level `delta_step`
uses supplied vectors directly; `stack_step` normalizes its recurrent vectors.

`attention_token` concatenates arrays to expose the cache operation clearly.
Production caches can reserve storage to avoid copying the old prefix at every
step. The examples count live persistent payloads, excluding old snapshots,
output traces, temporary workspaces and Python object overhead.
