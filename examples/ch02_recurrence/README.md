# Chapter 2: Recurrence and Encoder-Decoder Models

This example starts with a scalar recurrent update, works through one LSTM
memory update, and trains a small encoder-decoder on color reversal. Separate
one-layer LSTMs encode the source and decode the target. The final encoder
hidden and cell states initialize the decoder. There is no attention.

Run these commands from the repository root:

```sh
python -m pip install -r requirements-ch02.txt
python -m examples.ch02_recurrence.run_examples
python -m unittest discover -s tests -p 'test_recurrence.py' -v
```

The default command runs arithmetic and forward/backward checks without
training. Its randomly initialized predictions illustrate the generation
procedure, rather than the model's performance after learning.

## Following the numbers

`model.rnn_trace` reuses `h = tanh(0.5*h + x)` at every position. Starting from
zero, inputs `(0.2, 0.4, -0.1)` produce states approximately
`(0.197375320225, 0.461084444830, 0.129805707304)`. Reversing the input changes
the final state to `0.352470022360`.

`model.gated_cell_update` starts with already-computed gates. Previous cell
`0.5`, forget gate `0.8`, input gate `0.5`, candidate `-0.4`, and output gate
`0.6` give cell `0.2` and hidden value `0.118425192135`. These are hand-chosen
teaching values. `model.lstm_step` also calculates gates from input and hidden
vectors; tests compare it with an independent `torch.nn.LSTMCell` using the
same parameters. This is the modern LSTM with a forget gate, rather than an
exact reconstruction of the original 1997 cell.

## Source and target sequences

The fixed vocabulary is PAD `0`, BOS `1`, EOS `2`, followed by six color IDs
`3` through `8`. For source `(3, 4, 5)`, meaning **red green blue**:

```text
source:          red    green  blue
decoder input:   BOS    blue   green  red
target label:    blue   green  red    EOS
```

`data.collate` creates shifted rows and right-pads a batch. Source lengths
identify valid encoder positions. Packing the embeddings ensures that final
encoder states come after the last source token rather than after padding;
unsorted source lengths are accepted and the original batch order is restored.

| Value | Shape |
| --- | --- |
| `batch.source` | `[B, S]` integer IDs |
| `batch.lengths` | `[B]` source lengths |
| `batch.decoder_input`, `batch.target` | `[B, T]` integer IDs |
| Each final encoder state, hidden or cell | `[1, B, 64]` |
| Teacher-forced vocabulary logits | `[B, T, 9]` |
| Generated IDs | `[B, generated_steps]` |

Here B is batch size, S is padded source width, and T is padded target width.
The leading `1` in each state is the number of recurrent layers. Matching
encoder and decoder widths allow direct state transfer. Different widths or
layer counts would require an explicit mapping.

During training, `forward` receives correct preceding target tokens.
`token_loss` averages raw-logit cross-entropy over non-PAD labels, including
EOS. PAD remains a vocabulary class in each valid softmax denominator; only
positions whose label is PAD are ignored.

During generation, `greedy_decode` receives only the source and its lengths.
Call `model.eval()` first. It begins with BOS, feeds back its own selections,
includes EOS in the result, and pads already-finished rows while other rows
continue. A model that never emits EOS stops at `max_new_tokens`; reaching
that cap is not successful completion. Argmax considers all nine classes,
so an untrained model can also select PAD or BOS incorrectly.

## Training the small model

```sh
python -m examples.ch02_recurrence.run_examples --train --steps 750 --output output/ch02
```

The default run uses one CPU thread and seed 2026. It samples 192 training,
48 validation, and 48 test sequences; each split balances source lengths
three through six. No complete source tuple is shared between splits, although
all splits use the same six colors. Targets reverse the source and add EOS.
This checkable task does not model translation or language understanding.

Both embeddings have width 16 and each LSTM has hidden width 64. The model has
42,857 parameter-tensor entries and no dropout. This count includes two 16-entry PAD embedding rows that remain fixed at zero; it does not count the changing recurrent states. Adam uses learning rate `0.003`,
batch size 32, and gradient clipping at norm 1.0. Each update samples without
replacement within the batch. Validation is checked every 50 updates and at
the final update. The checkpoint minimizes validation token negative
log-likelihood; test results do not select it.

The checked CPU run (Python 3.12.14, PyTorch 2.15.0, NumPy 2.5.4) selected update 600. Validation NLL was
`0.106924302876` with 40/48 exact greedy sequences. Test NLL was
`0.090301297605` with 43/48 exact sequences: 12/12 at each source length three,
four, and five, and 7/12 at length six. These results describe one split and
initialization, not a general length limit or a comparison with a Transformer.

Evaluation reports both teacher-forced token accuracy and complete greedy
exact match, including EOS. The common generation cap is seven new tokens,
enough for six colors plus EOS. All 48 test generations in the reference run
ended in EOS; five still had an incorrect color sequence.

`output/ch02/metrics.json` records the configuration, environment, split hashes,
validation history, per-length results, and generated IDs.
`output/ch02/best.pt` contains the selected weights. These outputs remain local
and are ignored by Git. Exact floating-point values can vary across versions
and devices; the source and seeds specify the experiment.

Only the selected checkpoint is saved, not a checkpoint at every validation
step. To inspect a less-trained model, use an untrained model or run a shorter
experiment in a separate output directory:

```sh
python -m examples.ch02_recurrence.run_examples --train --steps 50 --output output/ch02-early
```

For the default experiment, this Python example reloads the selected model and
compares scoring with generation on one held-out sequence. Change the checkpoint
path to inspect the shorter experiment.

```python
import torch
from examples.ch02_recurrence.data import VOCAB, collate, make_splits, words
from examples.ch02_recurrence.model import RecurrentSeq2Seq, greedy_decode

checkpoint = torch.load("output/ch02/best.pt", map_location="cpu", weights_only=True)
model = RecurrentSeq2Seq()
model.load_state_dict(checkpoint["model"])
model.eval()
_, _, test = make_splits(seed=checkpoint["configuration"]["seed"])
batch = collate(test[:1])
with torch.no_grad():
    scores = model(batch.source, batch.lengths, batch.decoder_input)
generated = greedy_decode(model, batch.source, batch.lengths, max_new_tokens=7)
print("Reference:", words(batch.target[0]))
print("Teacher-forced choices:", [VOCAB[i] for i in scores.argmax(-1)[0].tolist()])
print("Generated:", words(generated[0]))
```

The teacher-forced choices use a correct reference prefix at every position;
they are not one free-running prediction path. All their positions are printed,
even if an early choice is EOS. The generated row uses only the source and its
own preceding selections, and ends at its first EOS.

## Checks and book listings

Tests cover hand-calculated values, NumPy/PyTorch LSTM agreement, disjoint
balanced splits, BOS/EOS alignment, packed source-padding invariance, decoder
prefix causality, finite gradients, analytic padding-aware loss, first-step
agreement between teacher forcing and greedy generation, mixed EOS stopping,
and the generation cap for a model that never emits EOS. A separate controlled
evaluation test checks token-weighted loss, per-length counts, and complete
sequence grading, including erroneous PAD or BOS before EOS and a missing EOS.

Seven book listings are exported from the runnable code:

```sh
python scripts/export_snippets.py --chapter ch02
python scripts/export_snippets.py --chapter ch02 --check
```

## Exercises and solutions

The [14 exercises](EXERCISES.md) progress from distinguishing states and weights
to recurrent arithmetic, gates, state transfer, shifted labels, padding, and
analysis of a training experiment. The final challenge separates greedy
selection from the most probable complete sequence.

[Worked solutions](../../solutions/ch02_recurrence/README.md) cover every
exercise in a written PDF and an executed notebook with its own PDF export.
The training-update listing connects the forward model to gradient calculation
and parameter updates.

Implementation references: [PyTorch LSTM](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LSTM.html)
and [Embedding padding behavior](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Embedding.html).
