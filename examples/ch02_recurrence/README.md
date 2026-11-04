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
42,857 learned parameters and no dropout. Adam uses learning rate `0.003`,
batch size 32, and gradient clipping at norm 1.0. Each update samples without
replacement within the batch. Validation is checked every 50 updates and at
the final update. The checkpoint minimizes validation token negative
log-likelihood; test results do not select it.

The local reference run selected update 600. Validation NLL was
`0.106928281486` with 40/48 exact greedy sequences. Test NLL was
`0.090296871960` with 43/48 exact sequences: 12/12 at each source length three,
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

## Checks and book listings

Tests cover hand-calculated values, NumPy/PyTorch LSTM agreement, disjoint
balanced splits, BOS/EOS alignment, packed source-padding invariance, decoder
prefix causality, finite gradients, analytic padding-aware loss, first-step
agreement between teacher forcing and greedy generation, mixed EOS stopping,
and the generation cap for a model that never emits EOS.

Six book listings are exported from the runnable code:

```sh
python scripts/export_snippets.py --chapter ch02
python scripts/export_snippets.py --check
```
