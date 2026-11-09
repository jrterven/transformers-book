# Measured training, learning, and cache figures

Chapter 6 includes three measured figures from its actual color-reversal
Transformer. They expose a first Adam update, learning on a fixed validation
example, and attention/cache storage during free generation. They use the
canonical Chapter 5 model and Chapter 6 training/decoding functions. No external
model weights or dataset are downloaded.

## Reproduce

From the companion repository root:

```sh
python -m pip install -r requirements-ch06-figures.txt
python -m examples.ch06_training.run_experiment --output output/ch06
python -m examples.ch06_training.trained_figures \
  --checkpoint output/ch06/best.pt \
  --output-dir output/ch06-figures
```

The figure script replays the 2,000-update recipe from the checkpoint metadata.
It uses `train_step`, the same seeded minibatch sampler, Xavier initialization,
Adam settings, and validation intervals as the experiment. Observation hooks
capture tensors and gradients; they do not replace the training operation.
The replay's final complete state must equal the supplied checkpoint exactly
before its cache is plotted. If software differences prevent this check,
regenerate the checkpoint and the figures in the same environment. The published
layouts expect the reference architecture and selected update 2000.

Generated outputs are ignored by Git. The output directory contains three PNGs,
`observations.npz` with complete captured arrays, and `TRAINED_FIGURES_RUN.json`
with the configuration, source/split/checkpoint/state hashes, validation history,
selected example, numerical checks, and software versions. The tracked
[run record](TRAINED_FIGURES_RUN.json) describes the published figures. Pixel
bytes can change across plotting-library versions; the record identifies the
actual outputs rather than assuming identical rendering everywhere.

## One actual update

`ch06_real_update.png` shows the first sampled training minibatch: 64 sources
and 356 nonpadding labels, including 64 EOS labels. The first panel averages
predicted probabilities over those same valid positions and compares them with
the empirical reference-label frequencies. These are two distributions over
the nine vocabulary classes, not a distribution over source positions.

For vocabulary bias `b_v`, the loss gradient is the difference between its
mean predicted probability and label frequency. Independent aggregation matches
the captured autograd gradient within `8.42e-9`. For EOS the values are:

```text
mean probability:   0.0827656028
label frequency:   64/356 = 0.1797752809
raw bias gradient: -0.0970096812
clipped gradient:  -0.0154802939
first bias change: +0.0010000020
```

Clipping uses the norm of the **whole** model gradient vector, approximately
`6.2666552986`, bringing it to approximately `0.9999998412`. The plot displays
only the nine readout-bias coordinates. Adam updates all model parameters;
its first-step bias changes match the corrected first-moment/second-moment
formula within `3.48e-9`. The panel's change axis multiplies changes by 1000.
This initial-step behavior does not imply that later Adam changes always have
equal magnitude or that every individual token prediction improves per step.

## Fixed validation probe through learning

`ch06_learning_snapshots.png` uses the **first length-six validation source**,
selected before replay or examining predictions. It is validation index 0:

```text
source:    red yellow blue black white red
reference: red white black blue yellow red <eos>
```

Snapshots at updates 0, 100, and 2000 display the probability of each correct
label under its supplied reference prefix. The columns come from different
prediction distributions and need not sum to one. The final probabilities
round to `1.000` but remain below one. The example's NLL values are approximately
`2.600573`, `0.766453`, and `0.000163425`.

The lower panel separately runs full-prefix free generation. It suppresses
PAD/BOS, supplies only generated tokens after BOS, and stops at the first EOS
or eight new tokens. At update 100, the model gets 5/7 reference-prefix choices
right yet generates `red black white blue yellow <eos>`: it swaps two colors
and stops early. Teacher forcing still grades all seven supplied positions,
even if an earlier prediction chooses EOS. Its complete prediction IDs are
recorded without stopping early. At update 2000, the generated answer matches
all seven reference labels.

The validation history is recomputed at all recorded intervals, and its
minimum selects update 2000. Neither test predictions nor test metrics choose
the probe or displayed update numbers. These measurements concern one probe;
the original chapter learning curve and full-split scores remain separate.

## Cache storage during real generation

`ch06_real_cache.png` uses the selected trained checkpoint on that same probe.
The decoder consumes BOS followed by the six generated colors. The last red
input selects EOS and stops; EOS is never consumed or cached in this trace.

The top panel is the native full-prefix self-attention map of decoder layer 1,
head 1, evaluated on these actual generated inputs. It is **target self-attention**,
not source cross-attention. Future entries are exactly zero. The first query's
unit weight follows from there being only one allowed key. Some other displayed
values round to zero; rounded display alone does not establish an exact zero.

The lower panel counts `numel * element_size` of each actual KV tensor across
both decoder layers. One consumed position adds 1024 bytes (1 KiB) of self-KV;
the six-position source contributes a constant 6144 bytes (6 KiB) of cross-KV.
At the last call, self-KV is 7 KiB and total KV is 13 KiB. These counts exclude
weights, encoder outputs, temporary tensors, allocation overhead, and total
process memory.

At every prefix length, the script compares cached logits against the final
full-prefix row before suppressing vocabulary entries. The largest absolute
discrepancy is approximately `1.43e-6`; the selected tokens also agree. Appending a token keeps
all older self-KV values and all static cross-KV values exactly unchanged.
The figure measures stored tensors, rather than claiming a speedup from their
sizes. The chapter's separate benchmark times equal work.

The recorded CPU environment uses one thread, Python 3.12.14, PyTorch 2.15.0,
NumPy 2.5.4, Matplotlib 3.11.2, and Pillow 13.0.0. The saved state agrees with the
checkpoint used for Chapter 5's trained-component figures.
