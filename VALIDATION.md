# Local validation

## Chapter 4

Chapter 4 was validated on 2026-11-04 with Python 3.12.14 and NumPy 2.3.5 on
macOS (arm64). NumPy was already installed; no dependency installation was
needed for these checks.

All commands below were run from the companion repository root using that
Python interpreter. All three commands completed with exit status 0.

| Command | Result |
| --- | --- |
| `python3 -m examples.ch04_attention.run_examples` | Demo and all embedded invariant checks passed. |
| `python3 -m unittest discover -s tests -v` | 14 tests passed in 0.025 seconds. |
| `python3 scripts/export_snippets.py --check` | All 6 generated snippets were current. |

The demo produced these values, rounded to four decimal places:

```text
Unmasked output:
[[0.8022, 0.7967],
 [0.5989, 1.2033],
 [0.7517, 1.0000]]

Causal output:
[[1.0000, 0.0000],
 [0.3302, 1.3395],
 [0.7517, 1.0000]]

Batch/head output shape:  (2, 2, 3, 2)
Batch/head weights shape: (2, 2, 3, 3)
Masked backward maximum absolute error: 5.682e-11
```

The tests cover hardcoded float64 numerical references, row-wise softmax,
large finite inputs, mask broadcasting and invalid masks, incompatible shapes,
causal prefix and future-perturbation invariance, unmasked permutation
equivariance, batch/head isolation, central finite differences for masked and
unmasked gradients, rank-4 gradients, and zero gradients for a masked key.

These results describe this local validation run.

## Chapter 5 and the combined suite

Validated on 2026-11-04 using Python 3.11.17, PyTorch 2.14.1, and NumPy 2.4.6
on macOS (arm64), with CPU float64 calculations. PyTorch was available in an
existing local environment; that environment was not modified.

| Command | Result |
| --- | --- |
| `python -m examples.ch05_transformer.run_examples` | All worked examples, model forward/backward, prefix/future checks, and source-padding checks passed. |
| `python -m unittest discover -s tests -v` | 32 tests passed, with no skips: 14 for Chapter 4 and 18 for Chapter 5. |
| `python scripts/export_snippets.py --check` | All 14 snippets current: 6 for Chapter 4 and 8 for Chapter 5. |

The seeded, randomly initialized model has 540 learned parameters. Its output
has shape `(2, 4, 12)`; encoder, decoder, and cross-attention weights have shapes
`(2, 2, 5, 5)`, `(2, 2, 4, 4)`, and `(2, 2, 4, 5)`. The valid-token illustrative
cross-entropy was `2.973556394289663`. This loss checks the example's execution;
it does not measure trained translation performance.

The first query's concatenated heads were
`(0.802224185360, 0.796663721961, 0.802224185360, 0.598887907320)`.
After the chosen output projection, the vector was
`(1.203336278039, 1.096107675621, 1.203336278039, 0.997219768301)`.
The hand-worked FFN returned `(1, 2, 0, 3)`.

An independent review compared combined causal/padding attention against
`nn.MultiheadAttention`; maximum output and weight errors were `1.11e-16`.
Six sampled end-to-end parameter gradients agreed with central differences to
maximum absolute error `3.37e-10`.

The exercise checks also reproduced the intended behavior. Removing the causal
triangle while retaining target key validity changed earlier logits by about
`0.467593`; restoring the mask reduced that difference to zero. Perturbing
padded source IDs with a fixed validity mask left outputs unchanged. Two-layer
encoder and decoder stacks used distinct parameter storage, passed prefix and
padding checks, and produced finite gradients.

### Detailed Chapter 5 review

A fresh review on 2026-11-04 checked all five figures, their captions, and all
nine exercises. Figure 5.1 now explicitly selects the first attention row
before its displayed one-row value mixture. The demonstration helper's fixed
batch, vocabulary, and padding assumptions are documented, and the causality
exercise explains how to compare logits while its assertions intentionally fail.

The new `test_stacks_and_gradients_match_torch_reference` uses two encoder and
two decoder layers, mixed right-padding, unequal source/target lengths, and
dropout disabled. It compares the computational stacks from their embedded,
scaled, positioned inputs through the vocabulary projection. All 62 matching
parameter-gradient tensors and both input gradients are checked with absolute
tolerance `1e-11`. In the independent review, the maximum forward error was
`5.55e-16` and the largest parameter-gradient error was `5.33e-15`.
The combined suite passed all 32 tests after these changes; all 14 snippets
remain synchronized. The model equations and forward/backward implementation
required no corrections.
