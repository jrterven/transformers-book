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

### Chapter 5 figure and exercise audit

A second independent audit executed every exercise using the public component
interfaces and separately calculated numerical references. All twelve tasks
were achievable. Removing the causal triangle while retaining target-key
validity changed earlier logits by approximately `0.467592678` in the seed-7
example; restoring the triangle reduced that difference to zero. Keeping the
source-validity mask fixed while changing padded IDs also gave zero difference
in target logits. Padded target logits were nonzero, and excluding their labels
with `ignore_index` matched a loss computed on the valid labels alone.

A model with two layers per stack passed the prefix, future, and padding checks,
had distinct corresponding layer parameters, and received finite gradients in
all 64 parameter tensors. The exercise instructions now specify float64,
evaluation mode for output comparisons, the actual mask-construction helper,
and the distinction between returned FFN results and a constructed FFN module.
Their progression is five foundational tasks, five applied tasks, and two
optional programming challenges. No executable example or exported listing
changed in this audit.

All five figures were inspected at 240 dpi and checked against independent
arithmetic. The multi-head diagram's selected-row label was clarified to name
the attention-weight row explicitly. The other four figure PDFs were unchanged.

The 19 Chapter 5 tests were run again and passed without skips.

### Chapter 6 teaching review

The dataset, training, token-weighted evaluation, and decoding algorithms passed
review without changes. A new deterministic `loss_sanity_example` constructs
two padded targets and uniform nine-class logits: seven labels contribute,
giving loss sum `15.380572041353538`, mean NLL `2.1972245773362196`, and
perplexity 9. Its regression test checks the expected count and arithmetic.
The CLI prints this check before training, and it can be imported separately.

The complete 2000-update CPU recipe was reproduced in a separate output folder.
Every deterministic record field matched the reference, including the split
fingerprints, all 21 validation history points, selected update 2000, validation
and test metrics, length-group results, examples, and cache agreement. All 65
checkpoint state tensors were bitwise identical. Validation NLL remained
`0.00009950750973075628`, test NLL `0.00017950770614499396`, and both splits
retained 64/64 exact generated sequences including EOS. Timings changed between
runs; the manuscript keeps its original measured record and curves.

All 91 repository tests passed without skips, including 19 Chapter 6 tests.
All 43 exported snippets were current, including nine for Chapter 6. The modified
CLI also passed a one-update smoke run with a different data seed. The checked
environment used Python 3.11.17, PyTorch 2.14.1, and NumPy 2.4.6 on a macOS ARM64
CPU with one training thread.

Independent exercise probes checked padded target alignment, logit gradients,
corpus aggregation, EOS and cap behavior, fixed-batch overfitting, checkpoint
selection, cache faults, length extrapolation, and fixed-work timing. KV memory
arithmetic gave 30 KiB self plus 9 KiB cross storage. The ordinary cached evaluator
is unsuitable for a deliberately noncausal model: that experiment now explicitly
uses manual teacher-forced NLL and uncached generation with consistent mask
helpers. In-range and extrapolation comparisons use one explicit output cap.
The twelve exercises progress from arithmetic to small programs, followed by
four optional experiments.

All five figures and captions were visually checked. The loss mask label,
current-position cache notation, cross-attention query path, and BOS/EOS labels
were clarified. The target-shift and measured learning-curve PDFs were unchanged.

### Chapter 6 detailed teaching and solution review (November 2026)

This revision uses Python 3.12.14 and PyTorch 2.15.0 on one CPU thread.
It is a fresh measured run, distinct from the historical environments above.
All 20 Chapter 6 tests passed. The runner now records and applies one cap of
eight generated tokens to validation, whole-test scores, and every per-length
report. A regression check verifies the same cap reaches every evaluation.
The benchmark records parameter dtype and device alongside its timing phases.

The 2,000-update recipe selected update 2,000: validation NLL 0.0004995695,
test NLL 0.0011071211, and 64/64 exact generated test answers including EOS.
The full numerical record is [reference_metrics.json](examples/ch06_training/reference_metrics.json).
Running that recipe again inside the solution notebook reproduced all 21
validation records. Timing values are separate measurements and can vary.

All 15 exercises now have complete explanations in both PDF editions and
32 sequentially executed notebook code cells. They include all four optional
investigations. The length experiment gets 64/64 in-range answers but 0/64
length-7/8 answers with the same checkpoint and cap 10; 54 longer answers
end early. The leakage comparison changes only the two imported mask bindings
inside restoring contexts. Whole-reference/prefix-only NLLs are 0.8531/0.8531
for the causal model and 0.8460/7.1246 for the altered model. These comparisons
use validation examples, fresh paired initializations, and uncached generation.

The nine exported snippets were checked against canonical source. The private
manuscript uses the same recorded run for its table and learning curve. PDFs
are generated reproducibly from the executed notebook and visually inspected.

## Chapter 7: GPT, BERT, and T5

The new `examples/ch07_pretraining/` module constructs three learning problems
from readable word-level arrays: shifted next-token labels, BERT-style selected
labels and 80/10/10 input treatments, and paper-style T5 span reconstruction.
The small models reuse Chapter 5's sinusoidal positions, ReLU blocks, and
post-norm paths. They demonstrate information permissions and objectives;
they do not reproduce the historical model configurations or pretraining.
The guide documents a separate BOS start, the terminal reconstruction sentinel,
hand-selected spans, the small ordinary-word replacement pool, and ignored
loss labels as teaching conventions.

All 104 repository tests passed without skips, including 13 new Chapter 7
tests. All 50 exported snippets matched their canonical source, including seven
new excerpts copied into the private manuscript tree. Tests cover alignment,
selection and replacement, span boundaries and shifting, loss gradients,
causal and bidirectional visibility, target causality, source padding, and
one optimizer update for each family.

The CPU example completed all three fixed-problem runs with 200 Adam updates
at rate 0.01, seed 72, one thread, width 16, two heads, FFN width 32, two layers
per stack, and float32. Mean NLL changed from 2.9864246845 to 0.0011647906 for
continuation, 2.8815746307 to 0.0004992431 for selected-token recovery, and
2.8922839165 to 0.0015371589 for span reconstruction. All selected labels
matched (3/3, 1/1, 8/8); the encoder-decoder also generated the exact
reconstruction including S2 and EOS. These are memorization and wiring checks,
not a comparison of pretraining quality or generalization.

With identical random weights, seed 71, evaluation mode, and float64, changing
a future token gave zero earlier-logit difference under causal permissions and
0.1950116795 under bidirectional permissions. Independent exercise probes also
confirmed zero earlier-logit change after a later decoder-input change and
invariance to appended source padding; changing a valid source token affected
the decoder. All twelve exercises were checked for feasible APIs, numerical
answers, and increasing difficulty. The first five cover arrays and arithmetic,
the next four inspect the code, and the last three are optional extensions.

The tracked `examples/ch07_pretraining/reference_run.json` preserves the
constructed arrays, readable predictions, measured results, environment,
settings, and validation counts. It was copied to the private source tree as
`data/ch07_reference_run.json`. The checked environment used Python 3.11.17,
PyTorch 2.14.1, and NumPy 2.4.6 on macOS ARM64. The four chapter figures were
rebuilt and inspected with their captions, using the same arrays as the code.

### Chapter 7 detailed audit

A separate review re-rendered all four figure PDFs at 240 dpi and inspected
physical pages 108--118 of the compiled chapter, matching every arrow, mask,
shifted label, sentinel, EOS marker, and caption. No figure correction was needed.
The complete 200-update CLI run reproduced the tracked reference `results`
exactly, including losses, selected predictions, and T5 free generation. All
104 repository tests passed without skips; all 50 exported snippets remained
current and matched the private manuscript copies. The algorithms, excerpts,
and reference measurements therefore remain unchanged.

The review clarified one exercise ambiguity. A random BERT replacement equal
to the original word and the unchanged-input branch can produce the same IDs
and labels. The replacement exercise now asks readers to reuse one seeded
generator across calls and log the sampled draw/branch in an experimental
copy. An independent seed-19 probe found both unchanged and random-equals-
original cases, confirming why output IDs alone cannot identify the branch.

The optional classifier now suggests a task compatible with the fixed
vocabulary, varies context independently of the color label, and asks for
disjoint sentence arrangements, independent copies of the same model/head
states, and fresh optimizers. The manuscript also defines pooling, explains
cross-entropy reshaping and ignored rows versus vocabulary classes, and
makes the second GPT input position's Python index explicit. All twelve
exercises were checked independently, including the loss arithmetic,
zero gradients at ignored logit rows, the source/target/padding probes,
and frozen versus fine-tuned backbone updates.

### Chapter 7 generation and exercise review

The one-update T5 run previously emitted `BOS` as a continuation. It now
uses the full-prefix greedy decoder from Chapter 6, which excludes `BOS`
and `PAD` from token selection. The one-update regression passed, and the
200-update run reproduced every reference result. All 105 tests passed
without skips, including 14 Chapter 7 tests. All 50 snippets matched the
canonical exports, including the seven Chapter 7 listings.

The revised GPT/BERT figure distinguishes the reference labels from the
vocabulary distribution. A new render confirmed distinct arrowheads at the
selected BERT state and a single highlighted GPT state label. The exercise
checks confirmed all three BERT treatments with the seed-19 setup and zero
change in earlier causal logits when the final word of the blue-car example
changes. The training exercise names the output fields used to compare
predictions with their targets.

## Chapter 8: From the Original Block to Modern Decoders

Run the worked component examples and decoder checks with the Chapter 5
PyTorch environment.

~~~sh
python -m examples.ch08_modern.run_examples --output output/ch08.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17, PyTorch 2.14.1, and NumPy 2.4.6,
all 118 tests passed, including 13 Chapter 8 tests. The exporter checked
all 57 listings. The Chapter 8 tests compare RMSNorm and rotary attention
with independent calculations, check low-precision normalization,
verify causal and padding masks, and test the residual path in the
forward pass and gradients.

The one-thread float64 demonstration with seed 81 reproduced the
hand calculations for RMSNorm, SwiGLU, and the shared RoPE shift.
The decoder returned logits of shape [2, 3, 17] and had 5,744
parameters. Changing a future token left the earlier logits equal.
Appending padding and shifting all positions by seven changed the valid
logits by at most 3.33e-16 and 4.44e-16, respectively. All gradients
were finite. One SGD update at learning rate 0.01 moved the mean loss
on the same five labels from 3.245774 to 3.197668.

Independent exercise probes verified the small-input RMS mean squares,
the SwiGLU result with known weights, the signed rotary dot product,
the whole-prefix and split-rotation comparison, the pre- and post-norm
input gradients, and the parameter count with a third block.

### Chapter 8 study and gradient audit

The four vector figures and the compiled pages were inspected against the
equations and executable examples. The revised captions distinguish the
standalone normalization example, the signed RoPE rotation, and LLaMA's
final stack normalization. Independent probes checked the numerical answers
and the Python steps for all 12 exercises.

A NaN injected by a backward hook previously reached the SGD update in
`one_update`. The update now raises on missing or nonfinite gradients before
changing any parameters. The regression test verifies exact parameter
equality after the rejected step. All 119 tests passed, including 14 for
Chapter 8. All 57 exports passed the consistency check, and the successful
CLI output matched the original reference record byte for byte.

## Chapter 9. Scaling Data, Compute, and Expert Capacity

```sh
python -m examples.ch09_scaling.run_examples --output output/ch09.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
```

On macOS ARM64, all 128 tests passed with Python 3.11.17, PyTorch
2.14.1, and NumPy 2.4.6. Nine tests cover Chapter 9, including
equal-compute allocations, stable routing, grouped expert outputs,
capacity limits, and residual bypass. All 61 listings matched the
canonical code. The Chapter 9 example JSON also matched byte for byte
in a NumPy-only environment with Python 3.12.14 and NumPy 2.3.5.

The synthetic sweep selected 10 billion parameters and 400 billion
training token presentations from the five tested allocations. The
capacity-three example requested (1, 5, 1, 1) expert slots and filled
(1, 3, 1, 1). Rows 5 and 6 received zero expert updates. The raw
balance term from the requests and mean probabilities was 1.4.

### Chapter 9 study and numerical audit

All four figures were checked in the compiled chapter against the equations
and Python output. The numerical exercises were run from their revised
instructions, including the eight-row fixture for the balance term and the
finer allocation sweep at both budgets.

The budget helper now converts NumPy integer counts before multiplying so a
large FLOP estimate retains its value. Routing chooses the largest logit
before rounding can make two probabilities equal. The chosen probability
still scales the expert output. A NumPy integer capacity is saved as a JSON
integer in the example report.

All 131 tests passed, including 12 Chapter 9 tests, and all 61 excerpts
matched the source. The CLI reproduced the published reference JSON byte
for byte. An independent row-by-row reference also agreed with dispatch,
gated outputs, and residual bypass in 120 generated cases.

## Chapter 10. Incremental Decoding and KV Representations

~~~sh
python -m examples.ch10_kv.run_examples --output output/ch10.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64, all 144 tests passed with Python 3.11.17, NumPy 2.4.6,
and PyTorch 2.14.1. All 13 Chapter 10 tests also passed with Python
3.12.14 and NumPy 2.3.5. The example JSON matched byte for byte across
the two environments. All 66 exported excerpts matched their source.

For four query heads and four, two, or one KV heads, the single-token
and chunked results agreed with a full causal pass within 4e-16.
The float64 cache arrays used 768, 384, and 192 bytes. The two MLA
evaluation orders gave the same weights and outputs in the fixed
example, with eight compact cache scalars versus 20 for expanded
content with shared rotary keys.

### Chapter 10 study and numerical audit

All four figures and captions were checked against the equations and
Python results. The numerical exercises were solved from the revised
instructions, including the mask shape, cache contents, dtype, and score
scale checks.

Grouped attention applies masks on the batch and query head axes before
regrouping the weights for the value product. Regression cases distinguish
batch-specific from head-specific masks. The examples reject nonfinite
projections and malformed cache inputs, check that chunks cover every
supplied row, and preserve the storage dtype across appends. The MLA
implementations reject complex projections, and the absorbed path
checks for nonfinite outputs.

All 151 tests passed. The 66 excerpts matched their source, and the CLI
reproduced the published JSON byte for byte. Independent numerical
references agreed across different batch sizes, head counts, cache
dtypes, mask shapes, and unequal key, value, and latent widths.

## Chapter 11. Long Context, Sparse Access, and Sequence Compression

~~~sh
python -m examples.ch11_long_context.run_examples --output output/ch11.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64, all 162 tests passed, including 11 Chapter 11 tests.
All 69 listings matched their sources. The saved JSON matched the
command output and the run in the NumPy-only environment byte for byte.

The eight-position grids contained 64, 22, and 34 allowed pairs.
Gathered attention selected zero-based positions 1 and 6 and returned
the worked output (0.75, 0.5). Gathering every allowed row agreed with
the masked dense reference. The gated scalar summary was 3.7.
The mixed-history example excluded a summary ending after the query,
then produced 5.0 from one selected summary and two recent raw entries.

### Chapter 11 study and numerical audit

The four diagrams were checked against the worked examples. The revised
captions identify the layers in the global relay, distinguish selection
indices from gathered vectors, and trace the overlapping raw and summarized
positions. The exercise prompts specify input shapes and the accounting
boundary for main-attention KV entries.

The examples now reject complex inputs before converting to float64.
Selection sorts index scores in their input dtype so that large signed
and unsigned integers keep their order. Independent reference calculations
agreed with 120 connectivity masks, 720 sparse reads, 300 pooling cases,
and 1,920 completed-prefix checks under changes to future features or gates.

All 165 tests passed, including 14 for Chapter 11. All 69 excerpts matched
their source. The command-line example reproduced the published JSON byte
for byte.

## Chapter 12. Efficient Execution

```sh
python -m examples.ch12_efficient.run_examples --output output/ch12.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
```

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 174 tests passed,
including nine for Chapter 12. All 72 excerpts matched their sources.
The saved JSON matched stdout and the run in the NumPy-only environment.

The two-tile calculation returned 17/6 to float64 accuracy. An extra
fully masked tile left the state unchanged. For the rectangular example,
the largest absolute differences from Chapter 4's dense reference were
`2.22e-16` without a mask and `1.11e-16`
with the causal mask. A separate scalar calculation checked 144 tiled
outputs and 294 intermediate row states.

The size example counts one score array. At two bytes per score, the
full 4096-by-4096 array occupies 32 MiB and a 64-by-128 tile occupies
16 KiB. These are storage calculations. The examples run on a CPU.

## Chapter 13. Structured State Spaces and Long Convolutions

~~~sh
python -m examples.ch13_state_spaces.run_examples --output output/ch13.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 185 tests passed,
including 11 for Chapter 13. All 77 excerpts matched their sources. The
saved JSON matched the command output and the NumPy-only run.

The scalar recurrence and direct convolution both gave (2, 1, 4.5, 2.25).
They also agreed on a two-coordinate system with a nondiagonal transition.
The largest FFT error in those two examples was 5.55e-16. In the padding
example, the length-six FFT gave the causal output (1, 1, 1, 2). At
length four, the tail folded into the first two positions.

Independent checks covered 64 scans and impulse responses using explicit
matrix powers, 36 convolution and gating cases using scalar lag sums, and
nine bilinear conversions using an eigenbasis calculation.

## Chapter 14. Selective State Spaces

~~~sh
python -m examples.ch14_selective_ssm.run_examples --output output/ch14.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 197 tests passed,
including 12 for Chapter 14. All 82 excerpts matched their sources. The
saved JSON matched stdout and the NumPy-only run.

The scalar state trace was (1, 1.08, 2.54, 2.5846). All three SSD
calculations gave (2, 1, 4, 4.25). The second chunk's local outputs
(4, 4) received (0, 0.25) from the carried state. In the five-position
case, changing retentions, a warm initial state, and a short final chunk
still gave agreement across all three forms.

Independent reference sums verified 96 SSD configurations and 528
chunk schedules, plus 24 diagonal selective scans and grouped prefixes.
A high-precision calculation checked the scalar retain and write
coefficients at nine step sizes.

## Chapter 15. Delta Rules and Hybrid Stacks

~~~sh
python -m examples.ch15_hybrids.run_examples --output output/ch15.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 206 tests passed,
including nine for Chapter 15. All 86 excerpts matched their sources.
The saved JSON matched stdout and the NumPy-only run.

The delta-rule read gave (2.5, 0.75) and the second state column was
(0, 1.5). A separate unit-key check with a zero write, alpha 0.8 and
beta 1.5 gave carry factors -0.4 along the key and 0.8 orthogonal to it.
At four tokens the all-delta, hybrid, and all-attention toy stacks
held 12, 24, and 48 history scalars.

Independent calculations checked 144 delta updates and 1,500 streaming
steps across 240 schedule cases. The comparisons covered projected
history, state sizes, and the causal boundary. The examples use
hand-chosen projections and gates and run on a CPU.

## Chapter 16. From Pretrained Weights to Task Behavior

~~~sh
python -m examples.ch16_adaptation.run_examples --output output/ch16.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and PyTorch 2.14.1, all 216 tests
passed, including ten for Chapter 16. All 91 excerpts matched their
sources. Repeated runs produced identical JSON.

The head-only run trained 8 of 17 parameters and left the encoder
unchanged. Full fine-tuning trained all 17 and changed its largest
encoder entry by about 0.468. Both runs classified all four held-out
synthetic points correctly after training.

The shifted labels scored Yes and EOS. The example returned completion
log-probability -1.7917594692 and mean supervised loss 0.8958797346.
The adapted, merged, and reloaded rank-one projection all returned
(7, 3). The DPO example gave a margin of 1 and loss 0.3132616875.
The tests checked zero gradients on ignored logit rows, adapter gradients
with the base frozen, and detached preference reference scores.

Independent reference calculations checked 54 response batches with varied
lengths, 36 LoRA configurations, and 80 batches of preference pairs.

## Chapter 17. Designing Reproducible Evaluations

~~~sh
python -m examples.ch17_evaluation.run_examples --output output/ch17.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 227 tests
passed, including 11 for Chapter 17. All 95 excerpts matched their
sources. Saved JSON, stdout, and the NumPy-only run were identical.

The split assigned six groups to three partitions with 8, 1, and 3
rows and no shared groups. Both classifiers reached 75% accuracy on
the twelve-example fixture, with recall of 0.50 and 0.75. The three
reliability points were (0.20, 0.25), (0.50, 0.50), and (0.80, 0.75).
Their example-level Brier score was about 0.198333.

With 20,000 paired draws at seed 17, the percentile interval for
accuracy B minus A was about [-0.42, 0.42] by row and [-0.56, 0.54]
by group. The grouped resampling draws complete groups and weights
the replicate by its sampled rows. The fixtures illustrate calculation,
with no claim about the performance of trained classifiers.

Independent checks covered 20 group splits, 80 binary metric cases,
100 reliability and Brier cases, and 48 row or group bootstrap cases.

## Chapter 18. Language Understanding

~~~sh
python -m examples.ch18_understanding.run_examples --output output/ch18.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 236 tests
passed, including nine for Chapter 18. All 99 excerpts matched
their sources. Saved JSON, stdout, and the NumPy-only run agreed.

The three selected rows pooled to (1, 1) and gave logits (0, 1.25).
The subword example scored six of ten token positions, with one
target per word. The predicted New span missed Harbor, giving word
accuracy 5/6 and exact entity precision, recall, and F1 of 0.5.
The tests checked invalid BIO handling and the optional repair.

Independent calculations checked 108 pooled heads, 100 first-piece
alignments, and 120 BIO tag streams.

## Chapter 19. Translation and Summarization

~~~sh
python -m examples.ch19_generation.run_examples --output output/ch19.json
python -m examples.ch19_generation.run_examples --train-step
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17, NumPy 2.4.6, and PyTorch 2.14.1,
all 247 tests passed, including 11 for Chapter 19. All 104 excerpts
matched their sources. Default JSON matched the NumPy-only run.

Greedy decoding returned A then EOS with probability 0.33. Beam
width two returned B then EOS with probability 0.38. A one-step
limit returned an unfinished prefix. The clipped unigram precision
was 0.5, and the single-pair ROUGE-L F1 was 0.6.

The small encoder-decoder produced logits of shape (2, 3, 11) and
scored five labels. Mean loss on the same batch changed from
2.45547187158 to 2.36915952961 after one CPU update. Independent
checks covered 60 ragged batches, 80 probability trees, and 245
token pairs for the overlap calculations.

## Chapter 20. Search, Retrieval, and Grounded Assistants

~~~sh
python -m examples.ch20_retrieval.run_examples --output output/ch20.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 256 tests
passed, including nine for Chapter 20. All 109 excerpts matched
their sources. Saved JSON, stdout, and the NumPy-only run agreed.

BM25 ranked P2 first and P1 second. The hand-chosen dense scores
were 0.8, 0.6, and 0 for P1, P2, and P3. Reranking the two lexical
candidates placed P1 first. With a budget of ten words, the input
contained P1 and the literal phrase "mobility desk." The price case
had no annotated evidence or answer in the fixture.

Independent calculations checked 48 lexical configurations, 18 dense
score and rank cases, and 20 packing, evidence, and reranking cases.

## Chapter 21. Image and Video Understanding

~~~sh
python -m examples.ch21_vision.run_examples --output output/ch21.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 265 tests
passed, including nine for Chapter 21. All 113 excerpts matched
their sources. Saved JSON, stdout, and the NumPy-only run agreed.

The 8-by-8 RGB image gave four rows of 48 patch features and a
five-by-six token array after projection, CLS, and position addition.
Two ten-parameter heads trained on 12 synthetic images. On eight
held-out images, ordered patch means gave accuracy 1.0 and mean loss
0.08948969. Sorted means gave accuracy 0.5 and mean loss 0.69314718.

Three frames with four patches each gave 144 joint attention pairs,
or 36 temporal and 48 spatial pairs in the patch-only divided count.

Independent calculations checked 54 patch and projection cases,
27 probe fits, and 16 video pair enumerations.

## Chapter 22. Object Detection and Segmentation

~~~sh
python -m examples.ch22_detection.run_examples --output output/ch22.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 275 tests
passed, including ten for Chapter 22. All 117 excerpts matched
their sources. Saved JSON, stdout, and the NumPy-only run agreed.

The box and mask IoUs were 1/3 and 1/2. The cost example matched
Q2 to A and Q1 to B, leaving Q3 unused at total cost 3. The scored
boxes gave true positive, false positive, true positive. The
single-threshold all-points AP was 5/6.

Independent calculations checked 120 box pairs, 120 mask pairs, 150
assignments, and 288 ranked detection evaluations.

## Chapter 23. Vision–Language and Multimodal Models

~~~sh
python -m examples.ch23_multimodal.run_examples --output output/ch23.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 284 tests
passed, including nine for Chapter 23. All 121 excerpts matched
their sources. Saved JSON, stdout, and the NumPy-only run agreed.

The two-pair logits were [[2, 1.2], [0, 1.6]]. The directional
losses were about 0.277501 and 0.319972, with mean 0.298736.
The retrieval fixture gave mean recall 0.75 at rank one and
1.0 at rank two in both directions. Queries without annotated
relevant items were excluded from those means.

The bridge weights were (0.25, 0.75), with context (0.5, 3) and
projected row (0.5, 3, -2.5).

Independent calculations checked 27 scaled similarity configurations,
20 contrastive batches, 64 retrieval settings, and 48 visual bridges.

## Chapter 24. Speech and Audio

~~~sh
python -m examples.ch24_speech_audio.run_examples --output output/ch24.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 294 tests
passed, including ten for Chapter 24. All 124 excerpts matched
their sources. Saved JSON, stdout, and the NumPy-only run agreed.

The four-sample example gave raw powers 2.25, 1.125, and 0 at
0, 2, and 4 Hz. Each generated clip gave a seven-by-129 power
array. The 500 and 1,000 Hz labels held for the clean and noisy
fixtures, and the zero clip was labeled silence.

The transcript pair required one substitution, one deletion,
and one insertion over five reference words, giving WER 0.60.
A one-word hypothesis against an empty reference returned one
insertion and an undefined clip-level rate.

Independent calculations checked 252 framing and transform
configurations and 961 transcript pairs.

## Chapter 25. Image and Video Generation

~~~sh
python -m examples.ch25_visual_generation.run_examples --output output/ch25.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 302 tests
passed, including eight for Chapter 25. All 127 excerpts matched
their sources. Saved JSON, stdout, and the NumPy-only run agreed.

The noisy diagonal grid moved from cumulative signal coefficient
0.36 to 0.64 and then 1. With the supplied exact noise it recovered
the clean grid. Increasing one noise prediction by 0.25 in both
updates left an error of one third at that coordinate. The guidance
fixture returned (1, 0.6) at scale two.

Independent calculations checked 60 forward mixtures, 60 guidance
settings, and 30 deterministic updates against separate formulas.

## Chapter 26. Models for Code

~~~sh
python -m examples.ch26_code_models.run_examples --output output/ch26.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17, all 309 tests passed,
including seven for Chapter 26. All 131 excerpts matched their
sources. Saved JSON, stdout, and the second Python run agreed.

All four checked-in candidates passed both smoke tests. On the
six-case suite, they passed four, six, five, and six cases.
Two passed the entire suite, giving pass@1 of 1/2 and pass@2
of 5/6. The infilling fixture placed the later function call
before the middle marker in the serialized prompt.

Independent calculations checked 728 pass@k settings by subset
enumeration and all four candidates over 2,025 interval pairs.

## Chapter 27. Time-Series Forecasting

~~~sh
python -m examples.ch27_time_series.run_examples --output output/ch27.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 319 tests
passed, including ten for Chapter 27. All 135 excerpts matched
their sources. Saved JSON, stdout, and the NumPy-only run agreed.

The fixture fit a drift of four on values before index 24. It
omitted origins 22 and 23 because their targets cross the split,
then scored six origins from 24 through 29. The seasonal-naive
MAE was (4, 4, 4) and the fitted drift MAE was (0, 0, 0).

The separate two-origin interval example gave MAE (0.5, 0.5, 1.5),
coverage (1, 1, 0.5), and mean width (2, 2, 2). Independent
calculations checked 96 window configurations, 60 seasonal
forecasts, and 80 interval cases.

## Chapter 28. Learning from Tables

~~~sh
python -m examples.ch28_tabular.run_examples --output output/ch28.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 327 tests
passed, including eight for Chapter 28. All 139 excerpts matched
their sources. Saved JSON, stdout, and the NumPy-only run agreed.

The four training rows gave median 4 and levels shop and web.
The missing-number, unseen-channel row encoded as (4, 1, 0, 0, 0, 1).
A held-out value of 100 and a held-out category left the fit fixed.

The weighted context example gave weights (1/6, 1/3, 1/2) and
positive-class probability 5/6. The binary Brier example gave 0.07.
Independent checks covered 80 fitted schemas, 80 context cases,
and 50 probability scores.

## Chapter 29. Scientific Sequences and Biological Structure

~~~sh
python -m examples.ch29_protein.run_examples --output output/ch29.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17 and NumPy 2.4.6, all 337 tests
passed, including ten for Chapter 29. All 143 excerpts matched
their sources. Saved JSON, stdout, and the NumPy-only run agreed.

ACAC and AACC had the same composition and different ordered-pair
frequencies. Masking the two marker rows and padding gave a pooled
residue vector of (1, 1). With distance threshold 5 and sequence
separation 2, the contact pairs were (1, 3) and (2, 4) in one-based
indices. The second distance equals the inclusive threshold.

Independent checks covered 120 sequence and pair counts, 32 masked
means, and 360 distance and sequence-separation settings.

## Chapter 30. Decisions and Embodied Action

~~~sh
python -m examples.ch30_decisions.run_examples --output output/ch30.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17, all 347 tests passed,
including ten for Chapter 30. All 146 excerpts matched their
sources. Saved JSON, stdout, and the second Python run agreed.

Rewards (1, -2, 3) gave returns-to-go (2, 1, 3). On the original
grid, the state-only policy chose R and scored -1. With target
+1 the conditioned policy chose D, R, R, U and scored +1. On the
shifted grid it chose D, R and scored -1. An unsupported initial
target stopped before the first action.

Independent checks covered 300 return sequences, 300 count-policy
fits, and 25,000 rollouts against a separate grid implementation.

## Chapter 31. What Changed, and What Remains?

~~~sh
python -m examples.ch31_perspective.run_examples --output output/ch31.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
~~~

On macOS ARM64 with Python 3.11.17, all 357 tests passed,
including ten for Chapter 31. All 148 excerpts matched their
sources. Saved JSON, stdout, and the second Python run agreed.

At lengths 8 and 16, the full causal counts were 36 and 136.
A three-event window gave 21 and 45. For the delivery histories,
the three-event window recovered the first code after one or
two route updates and lost it after four or eight. All four
rules returned the latest route.

Independent checks covered 32,790 memory comparisons and 5,200
causal pair-count configurations.

## Appendix references and final checks

The appendices use the existing attention, training, normalization,
KV accounting, state-space, diffusion, and pair-count examples. The
repository-wide run passed 357 tests. All 148 excerpts matched their
canonical source, and the corresponding manuscript listings matched
these excerpts byte for byte.


### Chapter 7 complete textbook review - November 6, 2026

This review expands the exercise sequence from 12 to 14, with matching public
statements, all explanations in both PDF editions, and 23 executed notebook
cells. The written edition now includes the programming investigations' measured
results as well as the conceptual answers. New exercises distinguish input
corruption, key validity, and loss selection, and compare prompting, frozen
features, and full adaptation. The T5 exercise includes unequal source and target
lengths. The manuscript uses row-vector classification notation consistently.

The canonical BERT helper now exposes an optional `return_trace=True` result
without changing its two-output default or consuming extra random draws.
`TinyTokenModel.encode_hidden` exposes final position vectors before the
vocabulary head. Sixteen Chapter 7 tests passed without skips, including tests
of both additions; all seven exported snippets match their canonical sources.
The notebook executed in order with no errors on one CPU thread using Python
3.12.14, PyTorch 2.15.0, and NumPy 2.4.6. A fresh 200-update run reproduced the
recorded losses and selected predictions; the same reference JSON is copied
into the private manuscript repository. The classifier starts randomly and is
explicitly distinguished from an experiment on pretrained transfer.

The chapter's five figures explain parameter reuse, attention permissions,
GPT/BERT target paths, the three masking decisions, and T5 span reconstruction
including the shifted decoder input. Figure source and manuscript remain in
the private repository. Both solution PDFs are reproducible from the executed
notebook using `solutions/ch07_pretraining/build_pdfs.py`.
