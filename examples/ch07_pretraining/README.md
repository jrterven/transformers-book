# Chapter 7: GPT, BERT, and T5

The same text can supply different questions. After reading `the red`, predict
`car`. Given `the [MASK] car stopped`, recover `red`. Given a sentence with two
missing spans, generate those spans in order. This directory turns each question
into input IDs, label IDs, a loss, and an inspectable information path.

Run from the repository root, using the PyTorch environment from Chapter 5:

```sh
python -m pip install -r requirements-ch05.txt
python -m examples.ch07_pretraining.run_examples
python -m examples.ch07_pretraining.run_examples --train --steps 200 --output output/ch07.json
python -m unittest discover -s tests -v
python scripts/export_snippets.py --check
```

The default command prints the constructed examples and a visibility check.
It does not train or download anything. `--train` adds a fixed-example
memorization check for each objective. All runs use one CPU thread; the
visibility probe uses seed 71 and float64, and each training check resets
seed 72 and uses float32. The defaults are width 16, two attention heads,
feed-forward width 32, two layers per stack, and dropout zero.

## Reading the code

`data.py` contains a 17-token teaching vocabulary. Words are split on spaces;
this keeps the examples short and does not reproduce subword tokenization.
`PAD=0`, `BOS=1`, `EOS=2`, `[MASK]=3`, and `S0`, `S1`, `S2` occupy IDs 4--6.
The remaining IDs are ordinary words. `IGNORE=-100` is a label-array marker,
not a vocabulary token. `token_names` prints ignored slots as `-`.

`causal_pairs(tokens)` shifts a sentence by one position. For `the red car
stopped`, inputs are `the red car` and labels are `red car stopped`. The first
word supplies context rather than a scored target; this deliberately short
example adds no BOS or EOS. To teach a termination rule, include an end token
in the original sequence before shifting it.

`bert_corrupt(tokens, selected, generator=None)` takes explicit prediction
positions. At each selected position it saves the original word as the label,
then chooses an input treatment: mask with probability 0.8, draw a random
ordinary word with probability 0.1, or retain the original with probability
0.1. All other labels become `IGNORE`. Selected positions are chosen separately
from their replacements. The chapter's four-word example selects position 1 by
hand, with generator seed 7 making that input `[MASK]`; it is not a 15% sample.

The original BERT recipe selected 15% of eligible positions and used the
80/10/10 treatment mixture. For clarity, this implementation accepts the
selected positions instead of sampling a corpus recipe, excludes special
markers from selection, and draws random replacements only from the ordinary
words in its small vocabulary. A random replacement can equal the original
word. The input ID alone therefore does not tell us whether a position was
selected. Labels record that decision.

`span_corrupt(tokens, spans)` accepts ordered, nonempty, half-open intervals,
with visible text separating spans. For the sentence
`the red car stopped near the old bridge today` and spans `[(1, 3), (6, 8)]`,
it returns three arrays:

| Array | Tokens |
| --- | --- |
| Encoder input | `the S0 stopped near the S1 today EOS` |
| Decoder input | `BOS S0 red car S1 old bridge S2` |
| Labels | `S0 red car S1 old bridge S2 EOS` |

Sentinels are dedicated IDs rather than sequences of characters. This small
builder supports one or two hand-chosen spans and a paper-style terminal new
sentinel. It does not sample the original T5 corruption rate or span lengths.
Some production preprocessors have different boundary conventions. Here
`BOS` is an explicit start marker inherited from Chapter 6. Actual T5 uses its
padding-token ID as decoder start and keeps that start position valid; replacing
BOS with PAD in this teaching model without adjusting its validity mask would
be incorrect.

`model.py` reuses Chapter 5's attention, sinusoidal position vectors, ReLU
feed-forward layers, and post-norm residual paths. `TinyTokenModel(causal=True)`
has causal self-attention and no encoder or cross-attention. With `causal=False`,
it permits both directions across the supplied input. `make_model("t5")`
returns the small encoder--decoder already built in Chapter 5. These are
educational models of information flow. They do not reproduce the original
families' learned positions, GELU/MLM heads, T5 normalization, relative position
biases, tied weights, or other model-specific details. Their weights start
random; the names identify the objective and broad arrangement being studied.

`selected_loss(logits, labels)` receives raw vocabulary logits and a label
slot per input position. It ignores `-100` label slots and averages over the
remaining labels. It raises an error for an empty selection rather than
returning an undefined mean. Padding should be marked `IGNORE` in a label array;
PAD remains a vocabulary class in each scored softmax denominator.

`run_examples.py` prepares all three examples. `prepared_examples()` returns
`{family: (inputs, labels)}`. `inputs` is a tuple so that `model(*inputs)` works
for both a single token stack and an encoder--decoder. Every token array is
batched with shape `[1, length]`. `objective_step` is the same short sequence
for every arrangement: clear gradients, calculate logits, select the loss,
backpropagate, and update parameters.

## Checking an information path

`visibility_probe()` makes two stacks with identical parameter values. One
uses causal attention; the other uses bidirectional attention. Changing the
last word of `the red car stopped` to `moved` leaves the logits at `red`'s input
position unchanged in the causal stack. The bidirectional stack can use that
later word, and its logits change. Both models have two layers, so the check
also catches a later layer accidentally exposing a future input. These are
logit differences in untrained models, not a test of reading comprehension.

The checked float64 CPU run produced a maximum absolute difference of 0 for
the causal stack and approximately 0.195012 for the bidirectional stack. Exact
nonzero values depend on the initial weights and numerical environment; the
important distinction is whether a path is excluded or available. The tests
also perturb a padding embedding, check that a T5 decoder cannot read later
decoder inputs, and confirm that its encoder source can influence predictions.

## Checking the training connection

`overfit_example(family, steps=200)` repeats the same fixed example using Adam
with learning rate 0.01. It measures loss before and after, then counts correct
selected labels. `selected_predictions` and `selected_targets` show the token
names at those graded positions, so the reader can inspect the answers rather
than only their count. There is no held-out split or claim about learning language.
The check asks whether the objective, attention paths, gradients, and optimizer
can work together. The fixed examples contain three GPT labels, one BERT label,
and eight T5 labels, so their losses are not a family ranking.

The checked 200-step CPU run gave:

| Objective | Initial NLL | Final NLL | Correct selected labels |
| --- | ---: | ---: | ---: |
| Next token | 2.986425 | 0.001165 | 3 / 3 |
| Selected masked token | 2.881575 | 0.000499 | 1 / 1 |
| Missing spans | 2.892284 | 0.001537 | 8 / 8 |

The complete numerical record is in [reference_run.json](reference_run.json).
It stores the environment, model dimensions, seeds, training settings, constructed
arrays, measured differences, and training results. To reproduce its `results`
field in the same environment, run:

```sh
python -m examples.ch07_pretraining.run_examples --train --steps 200 --output output/ch07.json
```

The output file contains the new measurements; the checked reference is kept
separate. On another numerical environment, compare loss values with a small
tolerance while checking the input arrays, selected-label counts, causal
invariance, and generated tokens directly.

The T5 teaching example additionally generates freely from BOS after training.
It reproduced `S0 red car S1 old bridge S2 EOS`. That check uses its own
previous choices rather than reference decoder inputs, but it still only
tests the one memorized example. The GPT demo scores a fixed continuation
without adding a termination token, and the BERT demo fills a selected slot;
neither is presented as an independently trained text generator.

To explore the chapter exercises, start with `prepared_examples()`, print the
arrays, and make a single change at a time. Keep the selected positions fixed
when studying the 80/10/10 treatments, use evaluation mode and identical weights
when comparing information paths, and report held-out performance separately
if you extend the fixed-example check into a real learning experiment.
