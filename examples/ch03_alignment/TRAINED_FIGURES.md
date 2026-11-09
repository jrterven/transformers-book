# Open the trained attention read

These three figures use the actual LSTM encoder-decoder trained by this
chapter. They show its learned scores and annotations, not invented values
or an image generator's interpretation. The task is synthetic color reversal.
No external dataset or pretrained checkpoint is downloaded.

Run from the repository root:

```sh
python -m pip install -r requirements-ch03-figures.txt
python -m examples.ch03_alignment.run_examples --train --steps 750 --output output/ch03
python -m examples.ch03_alignment.inspect_trained \
  --checkpoint output/ch03/best.pt --output-dir output/ch03-inspection
```

The inspection command evaluates the frozen selected checkpoint, then saves
three PNGs, `observations.npz`, and `TRAINED_FIGURES_RUN.json` in the output
directory. The manifest next to this guide records the reference run.
It contains settings, package versions, checksums for source code, model
parameters, the checkpoint file, raw arrays, and the displayed figures.
The parameter checksum is independent of checkpoint serialization.
Checkpoint-file bytes can differ between saves of identical parameters.

## One real read: scores, weights, and context

`ch03_trained_read.png` opens the EOS row of the existing learned alignment
map. The illustrated test source is chosen after evaluation by the same rule
as `alignment_example`: a correct length-six example with the most distinct
colors, ties in test order. It never selects the checkpoint. The reference
source at test index 11 is `blue blue yellow green white red`; greedy
generation emits `red white green yellow blue blue EOS`.

The EOS prediction is step 7. Its query is the preceding hidden state, and
its input token is the final generated `blue`. Raw additive scores become
six source weights; the weights mix the original 64-coordinate annotations.
Features 1 and 2 were chosen in advance for display, without searching for
interpretable or large coordinates. They are signed learned features with
no assigned semantic names.

Reference weights are approximately
`(0.340202, 0.132385, 0.141649, 0.143845, 0.121973, 0.119946)`.
The first two context coordinates are `(-0.332294, 0.516547)`.
The probability of EOS is `0.999599`, a different distribution normalized
over the vocabulary. A confident ending does not require a sharp source read.

## Remove only the mask

`ch03_trained_padding.png` uses the first length-three test source, selected
before examining the intervention: test index 0, `red white black`.
Three PAD positions extend storage to width six, but true length remains
three. Packing preserves the real encoder states; unpacking supplies exactly
zero PAD annotations. The query, cell state, preceding token, parameters,
and memory are held fixed at the EOS prediction, step 4.

The correct read masks PAD scores to negative infinity. The intervention
removes only that score mask for this one read. The raw PAD scores are not
zero: the query and scoring bias can produce a nonzero score even with a
zero annotation. Without the mask, PAD receives `0.773006` of the total weight.
Zero PAD annotations then imply, for all 64 context features,

```text
unmasked_context = (1 - total_PAD_weight) * masked_context
```

The program checks this identity against the measured tensors. The EOS
probability changes from `0.999507` to `0.998537`; both choices remain correct.
This diagnoses a diluted read. It is not a complete unmasked generation or
an estimate of the error rate of a different model.

## Change a token, then follow the timing

`ch03_trained_timing.png` starts from the same six-color example after
consuming BOS at step 1. Two branches copy the same memory, hidden state,
and cell state. At step 2, one receives normal input `red`, the other `blue`.
The current attention rows agree exactly, because this architecture reads
with the preceding hidden state before consuming the current token input.
Vocabulary probabilities can change immediately after the LSTM update:
the probability of `white` changes from `0.995837` to `0.267031`.

At step 3, **both branches receive `white`**, isolating the earlier input
change. Their preceding states now differ, so their queries and weights
can differ. In this checkpoint, the largest attention weight moves from
source position 4 to position 3; the maximum absolute weight difference is
`0.777167`. Both panels use a fixed 0–1 weight scale.

These are controlled branches with supplied inputs, not two independent
greedy generations. The result verifies the dependency order in this
implementation; a nonzero future difference is not guaranteed for every
possible parameter set or substitution.

## Inspect the full arrays

```python
import json
import numpy as np

arrays = np.load("output/ch03-inspection/observations.npz")
report = json.load(open("output/ch03-inspection/TRAINED_FIGURES_RUN.json"))
print(arrays.files)
print(arrays["eos_scores"], arrays["eos_weights"])
np.testing.assert_allclose(arrays["eos_contributions"].sum(0),
                           arrays["eos_context"], atol=2e-7)
print(report["padding"], report["timing"])
```

`open_read` reconstructs the scorer and compares the weights and context
against native model attention. It checks that every context coordinate
lies within the corresponding range of valid annotation coordinates.
The EOS read must also agree with the existing greedy alignment trace.
The script verifies zero PAD annotations, the padding dilution identity,
and exact equality of current attention in the token intervention.
All measurements use full precision; displayed values are rounded.

The existing alignment tests independently check hand-calculated scores,
masking, gradients, padding invariance, decoder causality, and greedy traces:

```sh
python -m unittest discover -s tests -p 'test_alignment.py' -v
python scripts/export_snippets.py --check
```

Together with the aggregate held-out results, these observations explain
what the model computes. Attention weights alone remain an incomplete
explanation of which source words caused a prediction.
