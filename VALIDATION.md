# Local validation

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
