# Chapter 4: self-attention in NumPy

Juan Terven and Diana Córdova

The phrase *the red car* gives us three positions to work with. We first follow
one position: its query compares with three keys, softmax turns those scores
into three weights, and the weights combine three value vectors. Then we do
the same calculation for every position using matrices. The numbers are chosen
for teaching; they are not learned word embeddings or measured language behavior.

The example also introduces a causal mask, batch and head axes, and an optional
manual backward pass. NumPy is the only external dependency. Calculations use
`float64` so that the numerical examples and finite-difference checks can be
reproduced without a training framework.

Run these commands from the companion repository root (`transformers-book/`):

```sh
python3 -m pip install -r requirements.txt
python3 -m examples.ch04_attention.run_examples
python3 -m unittest discover -s tests -p test_attention.py -v
python3 scripts/export_snippets.py --chapter ch04
python3 scripts/export_snippets.py --check --chapter ch04
```

The install command can be skipped when NumPy is already available. The runner
prints each stage rounded to four decimal places and checks the results at full
precision. No model download, accelerator, network access, or training is needed.

The [15 exercises](EXERCISES.md) progress from roles and hand calculations to
masking, tensor axes, permutations, and two optional gradient challenges.
[All solutions](../../solutions/ch04_attention/README.md) are available in an
executed notebook and two PDFs generated from that notebook.

`one_query_example(query_index=0)` in `run_examples.py` returns the scaled scores,
weights, separate weighted value contributions, and their sum. For the first
query, the contributions are approximately

```text
weight 0.4011 times value [1, 0] -> [0.4011, 0.0000]
weight 0.1978 times value [0, 2] -> [0.0000, 0.3956]
weight 0.4011 times value [1, 1] -> [0.4011, 0.4011]
                                      sum -> [0.8022, 0.7967]
```

The scores and weights have shape `(3,)`, the contributions have shape `(3, 2)`,
and the output has shape `(2,)`. The explicit loops are there to make each key
comparison and each value contribution visible. The runner compares this output
with the first row of the matrix calculation below. Choosing query index 1 or 2
follows another row through the same steps.

The three-token example uses

```text
H = [[1, 0, 1, 0],     Q = K = [[1, 0],     V = [[1, 0],
     [0, 1, 0, 2],              [0, 1],          [0, 2],
     [1, 1, 1, 1]]             [1, 1]]          [1, 1]]

unmasked output = [[0.8022, 0.7967],
                   [0.5989, 1.2033],
                   [0.7517, 1.0000]]

causal output   = [[1.0000, 0.0000],
                   [0.3302, 1.3395],
                   [0.7517, 1.0000]]
```

`attention.py` provides three functions:

| Function | Result |
| --- | --- |
| `stable_softmax(scores, allowed_mask=None)` | Probabilities along the last axis. |
| `scaled_dot_product_attention(q, k, v, allowed_mask=None)` | `(output, weights)`. |
| `attention_backward(q, k, v, upstream, allowed_mask=None)` | `(dQ, dK, dV)` for `upstream = dL/dOutput`. |

Inputs to attention have shapes `q: (..., Nq, Dk)`, `k: (..., Nk, Dk)`, and
`v: (..., Nk, Dv)`. The final two axes are tokens and features. Leading axes
must match exactly; `(batch, head, tokens, features)` is supported without
mixing examples or heads. Outputs have shape `(..., Nq, Dv)` and weights have
shape `(..., Nq, Nk)`. `swapaxes(-2, -1)` transposes only token and feature axes.

A mask is boolean and uses **`True` for allowed keys**. It must broadcast to the
score shape without adding axes. Examples include `(Nq, Nk)` shared by all
heads, `(batch, 1, Nq, Nk)` shared by heads in each example, or the full score
shape. Masked weights and their score gradients are exactly zero. A row with
no allowed key raises `ValueError` because it has no probability distribution.
The implementation also rejects nonfinite inputs, incompatible shapes, and
nonboolean masks.

The tests check fixed numerical references, one-query/matrix agreement, causal
prefix and future-perturbation invariance, and isolation between batch/head
slices. Reordering keys and values together, along with their mask columns,
preserves the read; reordering them independently would change their pairing.
A future-input test changes the third row of `H` and recomputes all three
projections. The first two causal outputs stay unchanged, while the unmasked
outputs can change. This assumes earlier input rows do not already carry future
information. A separate value intervention verifies
`new_output - output = weights[:, 2, None] * [2, -2]` when only the third value
changes from `[1, 1]` to `[3, -1]` and the weights remain fixed.
A separate check reproduces the combined padding/causal exercise: the padded
key receives zero weight, but its query still reads valid values and returns
`[0.5, 1.0]`. These are different rules, so excluding a key does not erase a query.

The backward function differentiates attention with respect to independent
`Q`, `K`, and `V`; it includes the `1 / sqrt(Dk)` factor and the mask. In
self-attention, when all three come from `H`, combine the paths as
`dH = dQ @ W_Q.T + dK @ W_K.T + dV @ W_V.T`. These examples do not add dropout,
an output projection, residual connections, normalization, or a training loop.

`run_examples.gradient_max_error(q, k, v, upstream, allowed_mask, step=1e-6)`
compares the backward pass with central differences. The optional finite,
positive step lets Exercise 14 compare truncation and rounding errors without
copying the gradient-check implementation. The helper leaves its inputs unchanged.
The recorded Python 3.12.14 / NumPy 2.5.4 run passes all 20 Chapter 4 tests and
checks all seven exported listings. With the demonstration's causal inputs,
the default-step maximum absolute error is approximately `5.682e-11`.

The book's listings are generated from marked sections in `attention.py` and
`run_examples.py`. Edit those canonical files, then run the exporter. Its
`--check` option reports missing or out-of-date snippets without writing.
Repeated exports preserve unchanged files. The seven generated files are
`projections.py`, `one_query_example.py`, `stable_softmax.py`, `attention_core.py`,
`batch_heads.py`, `causal_example.py`, and `backward.py` under
`snippets/ch04_attention/`.
These excerpts are for LaTeX listings and omit input-validation helpers; use
the complete example modules to run the code.

## Optional observations of a pretrained head

The [pretrained figure guide](PRETRAINED_FIGURES.md) follows the same operations
inside a pinned DistilGPT2 checkpoint: actual Q/K/V features, scaled scores,
causal weights, and weighted-value contributions. A separate pass changes
only the final token to test that the earlier reads remain unchanged.
The program checks all 12 heads in the first block against this chapter's
independent NumPy calculation and saves the complete observations and provenance.
The regular NumPy example remains independent of this optional model download.
