# Chapter 4: self-attention in NumPy

Juan Terven and Diana Córdova

These examples expose the projections, scaled scores, row-wise softmax, value
mixture, causal mask, and backward pass of attention. NumPy is the only external
dependency. The calculations use `float64` so that the printed example and
finite-difference checks can be reproduced without a training framework.

Run these commands from the companion repository root (`transformers-book/`):

```sh
python3 -m pip install -r requirements.txt
python3 -m examples.ch04_attention.run_examples
python3 -m unittest discover -s tests -v
python3 scripts/export_snippets.py
python3 scripts/export_snippets.py --check
```

The install command can be skipped when NumPy is already available. The runner
prints values rounded to four decimal places and checks causal invariance,
permutation equivariance, isolation between batches and heads, and the manual
backward pass. Tests use full precision, fixed random seeds, hardcoded reference
values, and independent central finite differences of a scalar loss. No model
download, accelerator, network access, or training is needed.

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

The backward function differentiates attention with respect to independent
`Q`, `K`, and `V`; it includes the `1 / sqrt(Dk)` factor and the mask. In
self-attention, when all three come from `H`, combine the paths as
`dH = dQ @ W_Q.T + dK @ W_K.T + dV @ W_V.T`. These examples do not add dropout,
an output projection, residual connections, normalization, or a training loop.

The book's listings are generated from marked sections in `attention.py` and
`run_examples.py`. Edit those canonical files, then run the exporter. Its
`--check` option reports missing or out-of-date snippets without writing.
Repeated exports preserve unchanged files. The six generated files are
`projections.py`, `stable_softmax.py`, `attention_core.py`, `batch_heads.py`,
`causal_example.py`, and `backward.py` under `snippets/ch04_attention/`.
These excerpts are for LaTeX listings and omit input-validation helpers; use
the complete example modules to run the code.
