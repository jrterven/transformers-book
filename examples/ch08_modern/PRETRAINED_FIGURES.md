# Looking inside a pretrained modern decoder

The four measured figures follow one token through real RMSNorm, SwiGLU,
rotary attention, and residual additions. They use a published SmolLM2
checkpoint alongside the small teaching decoder. The teaching examples and
their tests continue to work without downloading weights.

From the repository root:

```sh
python -m pip install -r requirements-ch08-figures.txt
python -m examples.ch08_modern.pretrained_figures \
  --cache-dir /path/to/huggingface/hub \
  --output-dir output/ch08-pretrained
```

The first run downloads the pinned public safetensors weights and tokenizer.
It uses `token=False`, `trust_remote_code=False`, and no uploaded user data.
The checkpoint is roughly 270 MB; allow additional cache and runtime memory.
After download, add `--offline` to use only the local hub cache. No model
weights or generated arrays are redistributed here.

## Model and fixed probe

The model is [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Its card labels the
weights Apache 2.0. The observed safetensors SHA-256 is
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
The fully loaded causal predictor has 134,515,008 parameters, with no missing,
unused, mismatched, or newly initialized keys.

The [pinned configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json)
specifies 30 blocks, width 576, intermediate width 1,536, nine query heads,
three key--value heads, head width 64, RMSNorm epsilon `1e-5`, rotary base
100,000, and a 49,152-entry vocabulary. The checkpoint stores BF16 values;
the inspected model performs CPU float32 inference. It is an actual learned
predictor, with different dimensions and key--value sharing from the chapter's
ordinary multi-head teaching decoder.

All choices are fixed before inspection: prompt `The red car stopped near
the bridge.`, block 1, query head 1 (array index 0), query position 2, key
position 1, rotary pair 0, and displayed feature indices 0--7. No sampling,
training, prompt search, or most-interesting-head selection occurs. Evaluation
mode, frozen parameters, one CPU thread, deterministic algorithms, eager
attention, and `use_cache=False` make the observation path explicit.

The eight decoded pieces are `The`, `·red`, `·car`, `·stopped`, `·near`,
`·the`, `·bridge`, and `.`. The middle dot denotes a preceding space, not an
extra character supplied to the model. The probe follows `·car` at index 2.

## Four figures

| File | Native observations and calculation |
| --- | --- |
| `ch08_pretrained_rmsnorm.png` | First eight input coordinates, rescaling before gain, learned gains, and native output; RMS over all 576 coordinates |
| `ch08_pretrained_swiglu.png` | Native gate/up projections, SiLU output, product, and cumulative down-projection contributions over all 1,536 intermediate features |
| `ch08_pretrained_rope.png` | Actual pair geometry, all 32 pair contributions to one score, and the native masked-softmax row over input keys |
| `ch08_pretrained_residual.png` | Two complete native additions in block 1; RMS at the same position after all 30 blocks and after the separate final norm |

The signed heatmaps have a common symmetric scale within each panel. Cells
show rounded values; all checks use unrounded arrays. Feature excerpts do
not replace complete-vector statistics or full-width matrix products.

## Checks and recorded values

RMSNorm is reconstructed in NumPy float64 from the complete input vector,
epsilon, and learned gains. At the probe its mean square is about 0.007032052,
and its stabilized denominator is about 0.08391693. The first output coordinate
is about -0.01988717. The maximum reconstruction error across the complete
first normalization output is below `5e-8` in the recorded run. RMS before
gain is about 0.999290 and after gain about 0.052170. Learned gains can be
negative; the output is not constrained to unit RMS.

SwiGLU's gate and up matrix products, stable sigmoid, SiLU, elementwise
product, and down projection are independently reconstructed. Native
intermediate modules are captured by forward hooks, including the SiLU
activation itself. At intermediate index 0, gate 0.707479 gives SiLU about
0.473900; multiplying by up value 0.307503 gives about 0.145726. Of all 1,536
activated gates, 218 are negative and ten exceed one. The first eight weighted
contributions to output feature 0 sum to about -0.072063, whereas the full
sum is about +0.153401. The cumulative curve is an ordered static sum,
not training time. Maximum independent FFN-output discrepancy is below `2e-5`
in the checked run.

RoPE uses the checkpoint's split-half layout: pair 0 comprises feature indices
0 and 32. The native rotary function is called on the captured projected
queries/keys and the exact cosine/sine arrays supplied during inference.
Independent trigonometric calculations check those vectors and their norms.
The rotations turn feature coordinates, rather than moving tokens. Pair 0
contributes about 0.764564; all 32 pairs give dot product about 12.131364 and
scaled score about 1.516420. This implementation's first three query heads
share key--value head 0; the reconstruction repeats the appropriate KV heads
before comparing all native first-block attention rows.

Independent masked softmax matches native attention within `3e-7` in this
run. Its row at `·car` is approximately `(0.275686, 0.129072, 0.595242, 0,
0, 0, 0, 0)`. All future entries in every head and block are exactly zero.
Mixing the unrotated native values and applying the output projection
reconstructs the native attention update within `1e-6`. Shifting every index
by seven preserves the independent float64 scores within `8e-15`; the full
native float32 logits differ by at most about `4.20e-5`. The script uses
explicit absolute tolerances. No new context is added by this relabeling.

Both float32 residual additions match their native arrays exactly. Forward
hooks capture the raw outputs of all 30 blocks separately, so the residual
curve does not accidentally substitute the final normalized state for the
last raw block output. RMS rises from about 0.083857 at the embedding to
36.409342 after block 30, then changes to 2.039263 under the separate final
norm and its learned gains. This is one token in one fixed forward pass,
not a loss curve or evidence of increased accuracy. An independent final
normalization also checks the native final hidden state.

Changing only the last input token leaves every earlier logit exactly
unchanged in the recorded run. The shift, causality, reconstruction, and
shape assertions run automatically before figures are written. The source
contains numerical tolerances for other floating-point environments. These
examples inspect computations, without comparing model quality or runtime.

## Reproducibility record

[`PRETRAINED_FIGURES_RUN.json`](PRETRAINED_FIGURES_RUN.json) records the
November 9, 2026 measurements, model revision, weight and source hashes,
software versions, dimensions, inputs, values, and checks. A fresh run writes
a new record, the four PNGs, and `observations.npz` with complete captured
vectors, first-block internals, every block's raw outputs, logits, and all
native attention heads. The compact record and source are versioned; the
generated archive and images remain local. Compare numeric checks when
reproducing with different libraries, since plotting and archive hashes can
change with those libraries.

The observed implementation is
[Transformers 4.57.1's Llama module](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py).
This optional inspection does not change the canonical teaching model's
adjacent-pair RoPE or its ordinary multi-head attention.
