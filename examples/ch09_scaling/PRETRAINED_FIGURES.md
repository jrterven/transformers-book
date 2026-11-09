# Following rows through a pretrained expert layer

Four measured figures connect the small NumPy bank to a real Switch
Transformer. They show router probabilities, one selected expert's complete
calculation, a controlled capacity change, and the difference between hard
requests, soft probabilities, stored weights, and selected weights. The
canonical arithmetic examples and tests need only NumPy and no downloads.

From the repository root:

```sh
python -m pip install -r requirements-ch09-figures.txt
python -m examples.ch09_scaling.pretrained_figures \
  --cache-dir /path/to/huggingface/hub \
  --output-dir output/ch09-pretrained
```

The first run downloads the pinned public checkpoint and tokenizer. Its
`pytorch_model.bin` is about 1.24 GB; allow additional cache and runtime
memory. This checkpoint publishes no safetensors file. Loading explicitly
uses PyTorch's restricted `weights_only=True` loader, PyTorch 2.6 or newer,
`token=False`, and `trust_remote_code=False`. No custom repository Python is
executed and no input text is uploaded. Add `--offline` after download to
use only the local hub cache. Weights are not redistributed here.

## Fixed model and observation

The model is [`google/switch-base-8`](https://huggingface.co/google/switch-base-8),
revision `92fe2d22b024d9937146fe097ba3d3a7ba146e1b`. Its card labels the
weights Apache 2.0. The observed weight-file SHA-256 is
`ff91705b718f692fa0c994a49d094154583db40b536e80f20f52e05498ff6856`.
`SwitchTransformersEncoderModel` loads all 307,846,272 encoder parameters,
with no missing or mismatched keys and no newly initialized encoder weights.
The checkpoint's 250 decoder/output-head keys are unused deliberately;
the script rejects unused keys from any other component.

The [pinned configuration](https://huggingface.co/google/switch-base-8/blob/92fe2d22b024d9937146fe097ba3d3a7ba146e1b/config.json)
has 12 encoder blocks, six sparse encoder feed-forward sublayers, width 768,
eight bias-free ReLU experts per sparse bank, intermediate width 3,072,
and per-expert capacity 64. The first sparse sublayer is in block 2 (native
array index 1). The encoder attends to the full input; this is not a causal
decoder inspection. The published checkpoint stores BF16 values and this
run computes in CPU float32, with frozen weights, evaluation mode, one
thread, and deterministic algorithms. There is no training or sampling.

All choices are fixed before inspection: text `The red car stopped near the
old bridge. The blue car moved toward the river.`, first sparse encoder
sublayer, position 2 (the first `car`), and displayed feature indices 0--7.
The unpadded input has 18 positions including punctuation and `</s>`.
The figures replace SentencePiece's boundary marker with a middle dot;
it does not denote an extra input character. Input positions use native
zero-based indices; printed expert labels use native indices plus one.

## Four measured figures

| File | What is measured |
| --- | --- |
| `ch09_pretrained_router.png` | All 18 native probability rows over eight experts; orange outlines identify original top-one requests |
| `ch09_pretrained_expert.png` | First eight up/ReLU coordinates and native expert output, gated update, original residual, and residual sum at the fixed probe |
| `ch09_pretrained_capacity.png` | Requests and admissions under published capacity 64 and a deliberate local reduction to three; exact overflow positions |
| `ch09_pretrained_balance_storage.png` | Pre-capacity request fractions and mean probabilities; actual stored, per-token, and complete-input distinct matrix-entry counts |

Router cells share one probability scale from zero to one. Each signed
expert panel has its own symmetric scale. Rounded excerpts are for reading;
checks use complete unrounded vectors. No expert, prompt, or feature is
selected by its observed visual appearance.

## Native checks and values

Forward hooks capture the first sparse sublayer's original residual,
normalized input, logits, mask, gates, grouped expert inputs and outputs,
gated updates, and residual output. The archive also contains router logits
for all six encoder sparse sublayers and the final encoder hidden state.

Independent float64 products reconstruct all first-router logits from the
full 768-coordinate inputs and learned classifier. Stable softmax checks
probabilities and the retained maximum gate. Every native original request
is admitted under capacity 64. The maximum logit discrepancy in the
recorded run is about `3.32e-6`, and the probability discrepancy is below
`4.73e-7`. The fixed `car` requests printed expert 5 with gate
`0.5147019029`, not a renormalized gate of one.

Independent full-width up projection, ReLU, and down projection check all
768 expert outputs at the probe, with maximum discrepancy about `2.05e-5`.
A native module replay captures up/ReLU coordinates; its multiplication
groups can differ from the batched encoder call, so comparisons use the
explicit float32 tolerances in the source. The first eight intermediate
coordinates are zero after ReLU, but only 2,921 of all 3,072 coordinates
are zero. The other 151 explain why the expert output can be nonzero.
The first output coordinate is `10.19901371`; its gated update is
`5.249451637`. The original residual `17.98398209` becomes `23.23343277`.
The full captured float32 residual addition matches the native array exactly.

Original request counts are `(0, 0, 2, 3, 5, 3, 2, 3)`. The raw balance
diagnostic is `8 * dot(f, P) = 1.228662581`, using pre-capacity request
fractions and mean probabilities. It agrees with the native auxiliary
helper when that helper is explicitly given the original request IDs. This
scalar is not a complete model loss or proof of balanced requests.

The capacity intervention changes only this router's limit to `ceil(18/8)
= 3`. It runs the native sublayer on its same captured input, without
rerunning the rest of the encoder. The original capacity is restored in a
`finally` block. The native cumsum mask is checked against input-order
admission; it does not rank requests by gate. Logits and gates remain
exactly unchanged. Admissions become `(0, 0, 2, 3, 3, 3, 2, 3)`, so positions
11 and 15 overflow. Their expert branches are exactly zero and residuals
exactly preserved. Overflow is `2/18`; occupied reserved slots are `16/24`.
Accepted outputs agree within float32 tolerances because grouped expert
matrix multiplications can regroup after rows are removed.

The native sparse module returns an expert-index tensor after filtering;
an all-zero overflow mask has argmax zero and can therefore appear to have
expert index zero. The inspection derives **original requests from logits
and probabilities before filtering**, and tracks admission separately.
It does not reinterpret that post-filter tensor as original requests.
Changing capacity leaves the pre-filter `f`, `P`, and balance diagnostic
unchanged. Padding is absent; the published configuration does not ignore
padding automatically in every router operation.

Parameter counts come from actual module tensors and are independently
checked against their shapes: one expert 4,718,592; router 6,144; stored
branch 37,754,880; one admitted token 4,724,736. Six experts are requested
by the complete input, touching 28,317,696 distinct matrix entries including
the router once. These counts include zeros and exclude normalization,
attention, embeddings, and other blocks. They do not count multiplications,
optimizer memory, timing, or a measured speedup.

## Reproducibility

[`PRETRAINED_FIGURES_RUN.json`](PRETRAINED_FIGURES_RUN.json) records the
November 9, 2026 run, exact revision, checkpoint/source hashes, software,
IDs, pieces, dimensions, checks, and numeric values. The script writes four
PNGs, a fresh JSON record, and `observations.npz` containing the captured
arrays. Only source and compact metadata are versioned in the public code
repository. Compare numerical checks across environments; plot and archive
hashes can change with library versions.

The inspected implementation is the official
[Transformers 4.57.1 Switch module](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/switch_transformers/modeling_switch_transformers.py).
The model and routing method are described in the original
[Switch Transformer paper](https://arxiv.org/abs/2101.03961).
