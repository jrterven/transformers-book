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

## Chapter 6: training and generation

Validated on 2026-11-04 using Python 3.11.17 and PyTorch 2.14.1 on macOS
(arm64), with CPU float32 training and one CPU thread. Cache-equivalence
tests also use float64. No packages, pretrained weights, or external datasets
were downloaded for the run.

| Command | Result |
| --- | --- |
| `python -m examples.ch06_training.run_experiment --output output/ch06` | Completed 2000 training updates, validation selection, checkpoint reload, held-out generation, and cache comparison. |
| `python -m unittest discover -s tests -v` | 47 tests passed with no skips: 14 for Chapter 4, 18 for Chapter 5, and 15 for Chapter 6. |
| `python scripts/export_snippets.py --check` | All 22 snippets current: 6 for Chapter 4, 8 for Chapter 5, and 8 for Chapter 6. |

The fixed vocabulary has six colors plus PAD, BOS, and EOS. The deterministic,
disjoint data split contains 512 training, 64 validation, and 64 test sequences.
Each split balances source lengths 3 through 6. Repeated colors are allowed.
Targets reverse the source colors and include EOS. Seeds for both model and
data were 2026.

The trained model has 167,625 parameters: width 64, four heads, FFN width 128,
two encoder layers, two decoder layers, independent source/target embeddings,
and an untied output projection. Dropout is 0.0. Matrices use Xavier-uniform
initialization. Adam uses learning rate 0.001 and betas `(0.9, 0.999)`;
gradient norms are clipped to 1.0. Each update samples 64 training sequences
without replacement within the batch. Validation is checked every 100 updates.
The checkpoint is selected by minimum validation token NLL, not test accuracy.

The final selected checkpoint was at update 2000:

| Metric | Validation | Test |
| --- | ---: | ---: |
| Non-padding target tokens, including EOS | 352 | 352 |
| Token NLL, natural logarithms | 0.0000995075097308 | 0.000179507706145 |
| Perplexity | 1.00009951246 | 1.00017952382 |
| Teacher-forced token accuracy | 352/352 | 352/352 |
| Greedy complete-sequence exact match, including EOS | 64/64 | 64/64 |
| Generated sequences ending in EOS | 64/64 | 64/64 |

These are results for one small synthetic split and initialization. They do
not measure translation, extrapolation to longer sequences, or robustness to
new tokens. The configuration, split hashes, full validation history,
predictions, and environment are recorded in local `output/ch06/metrics.json`;
the selected model is in `output/ch06/best.pt`.

The eight held-out examples checked by the demo generated the same IDs with
and without caching. Comparing cached next-token logits against full-prefix
logits over their supplied prefixes gave a maximum absolute float32 difference
of `4.76837158203125e-6`. The float64 test independently compares every prefix
through two decoder layers with source padding at tolerance `1e-12`.

The same run measured median CPU times over 25 repetitions, batch size 1,
one thread, and a fixed five-token supplied decoder prefix:

| Timed operation | Median milliseconds |
| --- | ---: |
| Source encoding | 0.204167 |
| Cached target prefill of the first four tokens, through serial steps | 0.786459 |
| Full five-token prefix to next-token logits | 0.402417 |
| Cached fifth token to next-token logits | 0.197625 |

Cached decode timing excludes source encoding and target prefill; it is not
end-to-end latency. Both next-token measurements compute logits at the same
prefix position. The cached reference grows tensors with `torch.cat` and its
prefill is serial. These small CPU timings are noisy implementation checks,
not claims about production serving or GPU speedups.

Both execution paths receive warmup calls. This timing example encodes four
source tokens without source padding. A second CLI smoke run with
`--steps 1 --data-seed 2027 --output output/ch06-smoke` verified that the
benchmark also handles a shorter first source and four-token decoder prefix.
That one-update run checks the command path, not learned accuracy.

The selected model's test diagnostics were also recorded separately at each
source length. Every length had 16 test sequences, all terminating with EOS
and all exactly correct. Their token NLL values were approximately
`0.0000369741` (length 3), `0.000618374` (length 4), `0.0000366073` (length 5),
and `0.0000699656` (length 6).

Chapter 6 tests cover split isolation and balance; exact BOS/EOS alignment;
analytic padding-aware cross-entropy and its gradient; token-weighted
evaluation across different batch partitions; full-prefix agreement with
the Chapter 5 forward pass; per-layer self-KV, absolute position offsets,
and static cross-KV with source padding; immediate EOS, mixed stopping times,
and maximum-length termination; saved checkpoint snapshots; and actual
learning and generation on a tiny fixed dataset.
