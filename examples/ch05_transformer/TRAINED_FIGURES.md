# Inspecting the trained encoder-decoder

The Chapter 5 figures measure the actual post-norm modules in the small
color-reversal model trained in [Chapter 6](../ch06_training/README.md).
They show learned cross-attention, residual addition followed by LayerNorm,
and a positionwise ReLU FFN. They are plots of captured model tensors, rather
than artist-drawn examples or generated illustrations.

The main Chapter 5 runner remains a separate, randomly initialized architectural
demonstration. This optional inspection uses a saved trained checkpoint and
takes no optimizer steps. No external model weights or dataset are downloaded.

## Reproduce the figures

From the companion repository root, install the optional plotting dependencies.
If you do not already have the Chapter 6 checkpoint, train it first:

```sh
python -m pip install -r requirements-ch05-figures.txt
python -m examples.ch06_training.run_experiment --output output/ch06
python -m examples.ch05_transformer.inspect_trained \
  --checkpoint output/ch06/best.pt \
  --output-dir output/ch05-inspection
```

The default training recipe uses one CPU thread, seeds 2026, 2,000 Adam updates,
batch size 64, learning rate 0.001, and validation every 100 updates. It selects
the checkpoint with the lowest validation NLL. It has two encoder layers, two
decoder layers, model width 64, four heads of width 16, FFN width 128, sinusoidal
positions, ReLU, post-norm ordering, and dropout 0.0. Chapter 6 documents the
training and validation procedure in full.

The output directory contains three PNGs, `observations.npz` with full arrays,
and `TRAINED_FIGURES_RUN.json` with checkpoint and parameter hashes, selections,
configuration, versions, numerical checks, and artifact hashes. The tracked
[run record](TRAINED_FIGURES_RUN.json) describes the published figures; generated
checkpoints and arrays stay in ignored output directories. Parameter values and
PNG bytes can differ across software versions; compare the recorded metadata
and numerical observations before expecting byte-for-byte reproduction.

## What is selected and supplied

The program selects the first length-four source from Chapter 6's held-out test
split **before examining its predictions or attention**. For the recorded run,
this is test index 0, `red yellow yellow blue`. Greedy generation produces
`blue yellow yellow red <eos>`, matching the reference. Generation suppresses
PAD and BOS and allows at most eight new tokens.

After generation, a separate teacher-forced observation pass supplies
`<bos> blue yellow yellow red` as decoder input, with the ordinary causal mask.
Attention-map row labels identify these **input positions**, not predictions.
The row for input `blue` prepares to predict `yellow`. Both occurrences of
`yellow` retain separate source/target positions. One successful generated
example is not an accuracy estimate or a translation benchmark.

All displayed components come from the first layer of their respective stack:

| Figure | Observation | Displayed subset |
| --- | --- | --- |
| `ch05_trained_heads.png` | Cross-attention in decoder layer 1; four heads and learned output projection | Maps for heads 1 and 2; all four projected contributions at decoder query 2, first 8 of 64 output features |
| `ch05_trained_postnorm.png` | Encoder source position 2: input, attention update, residual sum, standardization, learned scale/offset | First 8 of 64 features; statistics use **all 64** |
| `ch05_trained_ffn.png` | Same encoder position: learned `64 -> 128 -> 64` FFN | First 16 expanded features and first 8 output features; computation uses every feature |

The head outputs are **projected before being added**. The program slices
`W^O` into four `16 x 64` row-convention blocks. Each head's read times its
block yields a full-width contribution; their sum matches the cross-attention
module output. Signed contributions and nonnegative attention weights are
different quantities. The maps use a common 0–1 scale, and signed heatmaps use
a symmetric scale shared within each panel. Display rounding occurs after the
full calculation.

## Independent numerical checks

The checkpoint is frozen and evaluated on CPU. NumPy float64 calculations
reconstruct attention from captured float32 Q/K/V, head concatenation and
projection, population-variance LayerNorm, both FFN affine maps, and ReLU.
Assertions compare these calculations with the native module tensors. In the
recorded run, the largest absolute discrepancies were:

| Calculation | Maximum absolute discrepancy |
| --- | ---: |
| Attention weights | `1.19e-7` |
| Projected attention output | `3.32e-7` |
| LayerNorm output | `2.77e-7` |
| FFN output | `5.56e-7` |

The script also checks unit attention-row sums, verifies residual addition,
and perturbs only one FFN input row. The other FFN output rows remain exactly
unchanged. This last intervention concerns the **local FFN**, whose input rows
already contain context mixed by attention; it does not establish independence
of source positions in the whole encoder.

For source position 2, LayerNorm uses mean `0.5041652112`, population variance
`1.3353482012`, and epsilon `1e-5`. Its first coordinate changes from residual
`2.0052576065` to standardized `1.2989981596`, then to `1.4239535332` after
learned gamma/beta. The FFN has 53 positive ReLU coordinates out of 128 at this
position and first output coordinate `0.8463029861`. These measurements describe
this checkpoint and input, not universal properties of Transformer layers.

The recorded environment is Python 3.12.14, PyTorch 2.15.0, NumPy 2.5.4,
Matplotlib 3.11.2, and Pillow 13.0.0 on CPU. The run record includes hashes of
the inspection script and canonical model source.
