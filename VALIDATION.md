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

### Detailed Chapter 6 code review

A further review on 2026-11-04 checked the complete data, training, checkpoint,
generation, and cache implementations. The review corrected three behaviors:

- The data generator samples integer sequence indices instead of allocating
  every possible color tuple. All three recorded seed-2026 split fingerprints
  remain unchanged. A regression samples length-12 sequences from more than
  two billion possibilities while storing only the requested examples.
- Evaluation uses a common generation cap across all of its minibatches.
  Previously, a short sequence could receive a longer cap when batched with a
  longer sequence, changing termination rate when the batch size changed.
  The default is the dataset's longest source length plus EOS and one extra
  step, up to the model's positional limit. An explicit
  `generation_max_new_tokens` supports equal-cap comparisons across datasets.
  Regressions cover batch partitioning and the valid length-L-minus-one source
  case for a model with L decoder positions.
- Checkpoint loading preserves the saved parameter dtype. The float64 snapshot
  regression now checks restored logits at absolute and relative tolerance
  `1e-12`, rather than allowing a silent conversion to float32.

The updated suite passed all 50 tests without skips: 14 for Chapter 4, 18 for
Chapter 5, and 18 for Chapter 6. All 22 generated snippets are current. Loading
the existing selected float32 checkpoint and evaluating the unchanged splits
reproduced every recorded validation and test metric within `1e-12`, including
64/64 exact test sequences and NLL `0.000179507706145`. The training recipe and
recorded training curves were unchanged, so the 2000-update experiment was not
retrained for these fixes.
An additional one-update smoke run with `--data-seed 2027` completed dataset
creation, checkpoint selection, generation, cache comparisons, and the CPU
benchmark, including a shorter four-token decoder prefix.

## Chapter 1: token lookup and language-model scores

Validated on 2026-11-04 with Python 3.11.17, NumPy 2.4.6, and CPU float64
calculations. NumPy was already available; no environment changes, downloads,
training, or external datasets were needed.

| Command | Result |
| --- | --- |
| `python -m examples.ch01_language.run_examples` | Printed the lookup, ordered context, complete vocabulary distribution, batch shapes, and shifted targets; all embedded checks passed. |
| `python -m unittest discover -s tests -v` | All 58 tests passed without skips: 8 for Chapter 1, 14 for Chapter 4, 18 for Chapter 5, and 18 for Chapter 6. |
| `python scripts/export_snippets.py --check` | All 27 listings were current: 5 for Chapter 1, 6 for Chapter 4, 8 for Chapter 5, and 8 for Chapter 6. |

The hand-chosen embedding rows match the manuscript's original lookup figure.
For the ordered context **the red**, the hidden vector is `(0, 1, 0.5, 0.3)`;
logits are 2 for **car**, 1 for **red**, and 0 for all fourteen other vocabulary
entries. Softmax gives `P(car)=0.306506513543`, `P(red)=0.112757444918`, and
`0.041481145824` for each other entry. Correct-target negative log-probability
is `1.1825162724674187`. Reversing the context gives hidden vector
`(0.1, 0, 0.4, 0.9)` and a uniform distribution over sixteen vocabulary entries.
These are arithmetic examples with untrained weights, not language benchmarks.

The new tests check exact lookup values and repeated IDs; invariance to a
consistent ID/table renumbering; the hand-calculated hidden vector, logits,
and full-vocabulary normalization; token-order sensitivity; stable row-wise
softmax and additive score-shift invariance; BOS/next-token/EOS alignment;
stable negative log-probabilities; batch independence; and invalid IDs or
context widths. The complete previous chapter suite remained passing.

## Chapter 2: recurrence and the LSTM encoder-decoder

Validated on 2026-11-04 using Python 3.11.17, PyTorch 2.14.1, and NumPy 2.4.6
on macOS arm64. The experiment ran on one CPU thread. Existing dependencies
were used without changes to the environment.

| Command | Result |
| --- | --- |
| `python -m examples.ch02_recurrence.run_examples` | Scalar recurrence, LSTM gate arithmetic, model shapes, and forward/backward checks passed. |
| `python -m examples.ch02_recurrence.run_examples --train --steps 750 --output output/ch02` | Completed training, validation checkpoint selection, held-out generation, and metrics/checkpoint export. |
| `python -m unittest discover -s tests -v` | All 69 tests passed without skips, including 11 new Chapter 2 tests. |
| `python scripts/export_snippets.py --check` | All 33 listings current, including six for Chapter 2. |

The scalar update `h = tanh(0.5*h + x)` starts from zero and gives states
`(0.197375320225, 0.461084444830, 0.129805707304)` for inputs
`(0.2, 0.4, -0.1)`. Reversing the inputs gives final state
`0.352470022360`. The chosen LSTM gates give cell `0.2` and hidden value
`0.118425192135`. Tests independently compare the full NumPy gate calculation
with `torch.nn.LSTMCell`, using two examples, different input and hidden
widths, nonzero previous states, both PyTorch biases, and float64 arithmetic.

The trained model has independent width-16 source and target embeddings,
one width-64 encoder LSTM, one width-64 decoder LSTM, and a nine-class output
layer: 42,857 learned parameters. It transfers both final encoder states to
the decoder and has no attention or dropout. Source embeddings are packed
using their valid lengths so final states exclude right-padding.

The run used seed 2026, 192 training sequences, 48 validation sequences, and
48 test sequences. Each split balances lengths three through six; exact
source tuples are disjoint across splits. Targets reverse the six color IDs
and include EOS. Adam used learning rate `0.003`, batch size 32, default betas,
and gradient clipping at norm 1.0. Validation was checked every 50 updates.
The checkpoint was selected by minimum validation token NLL, at update 600
of the prescribed 750 updates.

| Selected-checkpoint metric | Validation | Test |
| --- | ---: | ---: |
| Non-padding target tokens, including EOS | 264 | 264 |
| Token NLL, natural logarithms | 0.106928281486 | 0.090296871960 |
| Teacher-forced correct tokens | 256/264 | 258/264 |
| Complete greedy exact sequences, including EOS | 40/48 | 43/48 |
| Generations ending in EOS | 48/48 | 48/48 |

Every sequence uses the same cap of seven generated tokens. Termination alone
does not imply a correct color sequence. Test diagnostics at each source
length were:

| Source length | Token NLL | Exact sequences | Ended in EOS |
| --- | ---: | ---: | ---: |
| 3 | 0.002764585428 | 12/12 | 12/12 |
| 4 | 0.025261895731 | 12/12 | 12/12 |
| 5 | 0.039257969707 | 12/12 | 12/12 |
| 6 | 0.230516508222 | 7/12 | 12/12 |

These results describe one small synthetic task and initialization. They do
not establish a general recurrent memory limit, performance on language, or
a comparison against the Chapter 6 Transformer experiment, which uses a
different training set and training recipe.

The local files `output/ch02/metrics.json` and `output/ch02/best.pt` retain the
configuration, split fingerprints, validation history, held-out predictions,
and selected weights. The experiment's test split SHA-256 is
`ac9dbe9dc4558ad77db6c0599ebd2f791e0405f05e5f28ba66a7f6d1ae644873`.

The new tests cover scalar arithmetic and token-order effects; LSTM gates;
independent NumPy/PyTorch LSTM agreement; disjoint balanced data; BOS/EOS and
right-padding alignment; packed states compared with individual sequences
in unsorted batch order; invariance to ignored source-padding values; causal
decoder prefixes and finite parameter gradients; first-step teacher/greedy
agreement; analytic loss and ignored-position gradients; mixed EOS stopping;
and an explicit generation cap when EOS is never predicted.

### Detailed Chapter 2 review

A separate review on 2026-11-04 checked the arithmetic, evaluation, checkpoint
selection, source padding, and the instructions supporting the exercises. The
runtime algorithm and training recipe did not need changes. Two scalar test
references were corrected to retain the full precision of the calculated
second and third states; their displayed chapter values remain unchanged.

A new controlled evaluation regression uses four source lengths with 14 valid
target positions in a 20-position rectangular tensor. Its manually assigned
correct-token probabilities distinguish token-weighted NLL from the average
of sentence means. It checks teacher-forced correct-token counts, per-length
losses and counts, and complete greedy sequence grading. A correct sequence
with padding after EOS passes; an extra BOS or PAD before EOS remains an
error; reaching the generation cap without EOS is not complete.

| Review check | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` | All 70 tests passed without skips, including 12 for Chapter 2. |
| `python -m examples.ch02_recurrence.run_examples` | All arithmetic, shape, and forward/backward checks passed. |
| `python scripts/export_snippets.py --check` | All 33 exported listings remained current. |
| README checkpoint-inspection example | Reloaded the selected model, reconstructed the held-out split, and printed reference, teacher-forced choices, and free generation successfully. |

Independent rescoring of the saved selected checkpoint using float64
`log_softmax` and gathered target probabilities gave validation NLL
`0.106928280011` and test NLL `0.090296873918`, agreeing with the recorded
float32 reductions within rounding. Independent single-sequence generation
reproduced every recorded prediction: 40/48 complete validation sequences and
43/48 test sequences, all ending in EOS. Extending the source storage with
additional non-PAD IDs while retaining the true lengths left both encoded
states exactly unchanged. The original trained checkpoint and recorded
experiment metrics were retained.

The chapter guide now explains that only the validation-selected `best.pt` is
saved. It provides a separate output path for a shorter training run and a
tested example for comparing teacher-forced scoring with free generation.
The teacher-forced display retains every reference prediction position, even
after an incorrect EOS choice; a constructed `[EOS, 3, 4, EOS]` score row
verified that all four choices are printed. Free generation still ends at
its first EOS.

## Chapter 3: learned recurrent alignment

Validated on 2026-11-04 using Python 3.11.17, NumPy 2.4.6, PyTorch 2.14.1,
and one CPU thread on macOS. The packages were already available; no
pretrained weights or external data were downloaded.

| Command or check | Result |
| --- | --- |
| `python -m examples.ch03_alignment.run_examples` | Given-score and additive-score arithmetic, masks, shapes, and finite forward/backward checks passed. |
| `python -m examples.ch03_alignment.run_examples --train --steps 750 --output output/ch03` | Training, validation selection, held-out evaluation, checkpoint writing, and attention-trace recording completed. |
| `python -m unittest discover -s tests -v` | All 83 tests passed without skips, including 13 new Chapter 3 tests. |
| `python scripts/export_snippets.py --check` | All 40 snippets are current, including seven Chapter 3 excerpts. |
| Reload selected checkpoint | Validation/test metric dictionaries and the entire saved attention trace reproduced exactly. |
| Chapter guide's inspection example | Printed the reference, all teacher-forced choices, generated tokens, and the 7-by-6 greedy attention matrix successfully. |

The first arithmetic example normalizes manually supplied log-probabilities
to weights `(0.05,0.10,0.85)` and returns context `(0.90,0.95)`.
The second uses separate hand-chosen additive-score parameters, giving
scores `(0.933322601637,-0.518922549974,0.117356736813)`, probabilities
`(0.596567504376,0.139623183989,0.263809311635)`, and context
`(0.860376816011,0.403432495624)`. Neither example is a measured translation.

The trainable model has 63,401 parameters: width-16 source/target embeddings,
a one-layer unidirectional width-64 encoder LSTM, width-32 additive attention,
a width-64 decoder LSTMCell receiving the 16-coordinate embedding plus
64-coordinate context, and a nine-class hidden-state projection. The encoder's
final hidden/cell pair initializes the decoder. This is a teaching adaptation,
rather than an exact implementation of Bahdanau's bidirectional GRU model.

The color-reversal sequences and split fingerprints exactly match Chapter 2.
Training uses 192 examples; validation and test use 48 each. Every split
balances source lengths three through six. Seed 2026 fixes data generation,
parameter initialization, and minibatch selection. Adam uses learning rate
0.003, default betas, batch size 32, and gradient clipping at norm 1.0.
Validation is measured every 50 updates. Minimum validation token NLL
selected update 750 of 750; the generation cap is seven new tokens.

| Selected-checkpoint metric | Validation | Test |
| --- | ---: | ---: |
| Non-padding labels, including EOS | 264 | 264 |
| Token NLL, natural logarithms | 0.001809378154576 | 0.001536062336527 |
| Teacher-forced correct tokens | 264/264 | 264/264 |
| Complete greedy exact match, including EOS | 48/48 | 48/48 |
| Generations ending in EOS | 48/48 | 48/48 |

Each test length has 12/12 exact generations. Test NLL at lengths three,
four, five, and six is approximately `0.000954897609`, `0.000909666123`,
`0.000979705830`, and `0.002792459447`. These are one synthetic split and
initialization. The model has more parameters and a different decoder input
than Chapter 2's example, so the results do not isolate attention's causal
benefit or measure translation performance or length extrapolation.

The illustrated greedy trace is test example 11:
`blue blue yellow green white red` produces
`red white green yellow blue blue EOS`.
The seven attention rows include the EOS prediction and six valid source
columns only. The maxima for ordinary colors follow source positions
6, 5, 4, 3, 2, and 1. The EOS row is broader; its largest weight is
approximately 0.3402 at position 1. This picture is an actual generated-path
trace, not a target-conditioned or manually assigned map. The example is
chosen after checkpoint selection among correct length-six test sequences
with the most distinct colors, using test order to break ties.

Tests cover independent scalar-coordinate score calculations, an explicit
NumPy/PyTorch additive-attention reference, stable source-axis normalization,
invalid masks, all-padding rejection, padding invariance of states/logits/
weights, zero gradient at masked annotations, gradient flow through scorer
parameters, teacher-forced causality, agreement between greedy attention and
its generated prefixes, batched versus single-sequence generation, EOS-row
retention and zero rows after termination, and independent token-weighted
evaluation with complete-sequence grading. The existing Chapter 1, 2, 4, 5,
and 6 suite remained passing.

### Chapter 3 review

The follow-up review found a NumPy shape error: passing the scoring direction
as a column could leave a singleton score axis and normalize over that axis.
The scorer now accepts both `[A]` and `[A, 1]` and returns one scalar per source
position. The weighted read rejects mismatched source lengths and incompatible
batch dimensions. Independent expected scores and contexts cover both direction
shapes, shared annotations, and batched annotations.

Attention-trace metadata now records whether a correct length-six example was
available and whether generation actually ended in EOS. Controlled failed runs
cover the fallback example and a generation cap reached without EOS. Reloading
the reference checkpoint preserved its numerical metrics and attention weights;
no retraining was needed.

All 86 tests passed without skips, the Chapter 3 demonstration passed, and all
40 exported snippets were current. All twelve exercises were checked against
the available model interface and arithmetic. Additional direct checks confirmed
source-padding invariance, the attention query's one-step timing, and joint
annotation/mask permutation invariance within floating-point precision. The
manuscript review also clarified the scorer's addition node and the exercise
that compares attention timing with the recurrent update.

### Chapter 4 teaching review

The existing scaled dot-product attention core and manual backward pass passed
review without algorithm changes. The demonstration now follows one query
with explicit key-comparison and value-contribution loops before calculating
all queries with matrices. Its seven exported excerpts include this new
`one_query_example` function. Fixed expected contributions and every query's
agreement with the matrix implementation are tested.

Additional checks preserve an output when keys, values, and mask columns are
permuted together, and reproduce the combined causal/padding exercise. The
padded key receives zero weight while its query still returns `[0.5, 1.0]`.
These checks help distinguish key availability from query validity.

On a macOS ARM64 CPU with Python 3.11.17 and NumPy 2.4.6, all 89 repository
tests passed without skips, including 17 Chapter 4 tests. All 41 exported
snippets were current. The Chapter 4 runner passed, with maximum absolute
causal finite-difference gradient error `5.682e-11`.

All twelve revised exercises were independently checked. The optional
projection-gradient exercise matched central differences to approximately
`1.8e-10` maximum absolute error across the four differentiated arrays. The
manuscript now separates this task from finite-difference step selection and
connects each matrix multiplication to the scalar sums it performs. The four
teaching figures were revised and visually checked with their captions.

### Chapter 5 teaching review

The attention projections, head ordering, masks, post-norm layers, and full
encoder-decoder backward path passed review without changes to `model.py`.
The runner now includes a separate hand-chosen vocabulary readout: the state
`(1,-1,0,2)` and a four-by-three projection produce logits `(2,1,0)` for
`el`, `coche`, and `EOS`. An independent expected-value test checks the logits,
probabilities `(0.665240955775,0.244728471055,0.090030573170)`, and cross-entropy
`1.407605964444` when `coche` is the reference. This small readout is separate
from the complete twelve-entry, randomly initialized teaching model.

With Python 3.11.17, PyTorch 2.14.1, and NumPy 2.4.6 on a macOS ARM64 CPU,
all 90 repository tests passed without skips, including 19 Chapter 5 tests.
All 42 exported snippets were current, including nine for Chapter 5. The
complete demonstration passed; its original illustrative cross-entropy
remained `2.973556`. No training or optimizer step was added.

The manuscript review made head split/merge ordering explicit, distinguished
embedding scaling from attention-score scaling, explained the encoder memory
and vocabulary readout, and expanded practice to twelve exercises. Independent
arithmetic checks confirmed the multi-head example, positional vectors,
parameter counts, and existing exercise results. The five figures and captions
were reviewed for numerical accuracy, arrow directions, and readability.
