# Chapter 3: Learning to Align

This example adds a learned source-reading operation to Chapter 2's recurrent
encoder-decoder. It calculates a score for every valid source position, turns
scores into weights, and reads a weighted mixture of encoder states.
Separate one-layer LSTMs still encode and decode the sequences.

Run from the repository root:

```sh
python -m pip install -r requirements-ch03.txt
python -m examples.ch03_alignment.run_examples
python -m unittest discover -s tests -p 'test_alignment.py' -v
```

The default command checks arithmetic, masks, shapes, and a forward/backward
pass. Its weights are untrained; it does not measure translation quality.

## Two numerical calculations

The annotations are the teaching vectors `(1,0)`, `(0,1)`, and `(1,1)`.
Manually supplied scores `log(0.05)`, `log(0.10)`, and `log(0.85)` give
weights `(0.05,0.10,0.85)` and context `(0.90,0.95)`.

A second example calculates scores from explicit hand-chosen parameters:
query `(0.2,-0.1)`, identity annotation and query projections, direction
`(1,-1)`, and zero bias. Its results are approximately

```text
scores:   ( 0.933322601637, -0.518922549974, 0.117356736813)
weights:  ( 0.596567504376,  0.139623183989, 0.263809311635)
context:  ( 0.860376816011,  0.403432495624)
```

These are separate calculations. The first score row is not the output of
the second example's parameters. Neither is measured language alignment.

## The recurrent model

`AttentiveSeq2Seq` uses width-16 embeddings, a width-64 encoder LSTM,
width-32 additive attention, a width-64 decoder LSTMCell, and a nine-class
output projection: **63,401 learned parameters**.

Packing excludes right-padding from recurrent updates; unpacking retains one
annotation per source position. Both final encoder states initialize the
decoder. This is a teaching adaptation: Bahdanau's original system used a
bidirectional recurrent encoder and GRUs, rather than this single-direction
LSTM. Here an annotation contains its source input and preceding context,
not a separate right-to-left summary.

At each step, attention compares the **previous decoder hidden state** with
the annotations. The source projection is computed once and reused. Invalid
source scores become negative infinity before softmax, so valid rows sum to
one and padding gets exactly zero weight. All-invalid sources are rejected.

The context and previous-token embedding enter the decoder LSTMCell together.
Its updated hidden state produces next-token logits. The attention read
happens **before** that update: the embedding supplied in the current call
affects the updated state, not the already-calculated attention row.

| Value | Shape |
| --- | --- |
| Source IDs and true lengths | `[B,S]`, `[B]` |
| Encoder annotations | `[B,S,64]` |
| Projected annotations | `[B,S,32]` |
| Source-valid mask | `[B,S]` Boolean |
| Each decoder state, hidden or cell | `[B,64]` |
| One context and attention row | `[B,64]`, `[B,S]` |
| Teacher-forced logits and weights | `[B,T,9]`, `[B,T,S]` |
| Greedy tokens and weights | `[B,T_generated]`, `[B,T_generated,S]` |

`encode(source,lengths)` returns `(memory,state)`. Memory contains
`(annotations,projected_annotations,source_valid)`; state contains
`(hidden,cell)`. The encoder first returns states `[1,B,64]`. Taking its one
layer gives the `[B,64]` pair used by `LSTMCell`.

`forward(source,lengths,decoder_input)` returns `(logits,attention)` and
receives correct preceding reference tokens. Target-padding labels are
ignored by loss, but their attention rows are still calculated. Exclude
those rows when displaying meaningful prediction steps.

`greedy_decode(model,source,lengths,max_new_tokens)` receives no target.
Call `model.eval()` first. It returns selected tokens and the attention used
to make each selection. The EOS prediction has a valid row. Later storage
positions contain PAD and zero attention; they are not additional reads.
Without EOS at the cap, generation is incomplete. Selecting PAD or BOS
before EOS is an error, not a stopping condition.

## Training and inspecting an alignment

```sh
python -m examples.ch03_alignment.run_examples --train --steps 750 --output output/ch03
```

The task, vocabulary, and split generator are reused from Chapter 2: 192
training, 48 validation, and 48 test sequences, balanced over lengths three
through six. Complete sources do not cross split boundaries. Identical seeds
give identical split fingerprints. This does not test length extrapolation.

Training uses seed 2026, one CPU thread, minibatches of 32, Adam at `0.003`,
no dropout, and gradient clipping at norm 1.0. Validation is measured every
50 updates and at the final update. Minimum validation token NLL selects
the checkpoint. Seven generated tokens allow six colors plus EOS.

The checked run selected update **750**. Validation NLL was `0.001809378155`
and test NLL `0.001536062337`. Greedy exact match was **48/48** on each split,
including EOS; each test-length group had 12/12 correct sequences. These
are one small synthetic run. More parameters and a different decoder input
make this an uncontrolled comparison with Chapter 2's no-attention model.

Local outputs, ignored by Git:

- `output/ch03/best.pt` contains selected weights and configuration.
- `output/ch03/metrics.json` records settings, versions, split hashes,
  validation history, per-length results, predictions, and an alignment trace.

`alignment_example` is a **greedy** trace including EOS and valid source
columns only. Its illustration is chosen after evaluation: among correct
length-six test examples, prefer more distinct colors, breaking ties by test
order. If none is correct, display the first length-six example and errors.
This illustration never selects the checkpoint.

The reference illustration is test example 11: **blue blue yellow green white
red** generates **red white green yellow blue blue EOS**. Attention maxima
move through source positions 6, 5, 4, 3, 2, and 1 for the colors. The EOS row
is broader and has no unique correct source-word alignment. Repeated words
occupy different positions. A weight map is not a complete causal explanation.

Reload and inspect without retraining:

```python
import torch
from examples.ch02_recurrence.data import VOCAB, collate, make_splits
from examples.ch03_alignment.model import AttentiveSeq2Seq, greedy_decode

checkpoint = torch.load("output/ch03/best.pt", map_location="cpu", weights_only=True)
cfg = checkpoint["configuration"]
model = AttentiveSeq2Seq(embedding_width=cfg["embedding_width"],
                       hidden_width=cfg["hidden_width"],
                       alignment_width=cfg["alignment_width"])
model.load_state_dict(checkpoint["model"])
model.eval()
_, _, test = make_splits(seed=cfg["seed"])
batch = collate(test[11:12])
with torch.no_grad():
    logits, teacher_weights = model(batch.source, batch.lengths, batch.decoder_input)
tokens, greedy_weights = greedy_decode(model, batch.source, batch.lengths, 7)
print("Reference:", [VOCAB[i] for i in batch.target[0].tolist()])
print("Teacher-forced choices:", [VOCAB[i] for i in logits.argmax(-1)[0].tolist()])
print("Generated:", [VOCAB[i] for i in tokens[0].tolist()])
print("Greedy attention:", greedy_weights[0])
```

Keep every teacher-forced choice, even after an early EOS prediction: these
rows use separate reference prefixes. Free generation stops at its first
EOS. To inspect errors, use an untrained model or fewer updates with a
separate output path. Only the selected checkpoint is saved.

## Checks and exported listings

Tests cover independent score calculations, NumPy/PyTorch agreement,
stable softmax, source masks, all-invalid sources, padding invariance,
zero gradients for masked annotations, decoder causality, scorer gradients,
greedy-prefix consistency, batched versus separate generation, EOS-aligned
rows, and independent token-weighted evaluation.

Seven excerpts are exported from the runnable source:

```sh
python scripts/export_snippets.py --chapter ch03
python scripts/export_snippets.py --check
```
