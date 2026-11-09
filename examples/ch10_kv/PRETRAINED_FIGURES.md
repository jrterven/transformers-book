# Real decoder caches and equivalent replay

The optional inspection uses published SmolLM2 and GPT-2 weights to observe
cache growth, grouped reads, complete-decoder equivalence, and a controlled
positional mistake. The regular NumPy examples remain offline and unchanged.

From the repository root:

```sh
python -m pip install -r requirements-ch10-figures.txt
python -m examples.ch10_kv.pretrained_figures \
  --cache-dir /path/to/huggingface/hub \
  --output-dir output/ch10-pretrained
```

The first run downloads pinned public safetensors checkpoints and fast
tokenizers (roughly 270 MB and 550 MB for weights, plus runtime memory).
The loader uses `token=False`, `trust_remote_code=False`, and
`use_safetensors=True`; no custom repository Python is executed and no input
text is uploaded. Add `--offline` once the local hub cache is populated.
Model weights are not redistributed in this repository.

## Models and fixed experiment

| Model | Revision | Native cache architecture |
| --- | --- | --- |
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 30 layers, 9 query heads, 3 KV heads, width 64; GQA |
| [GPT-2](https://huggingface.co/openai-community/gpt2) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 12 layers, 12 query/KV heads, width 64; MHA |

The cards label the weights Apache 2.0 and MIT, respectively. Weight-file
SHA-256 hashes are recorded in [the run record](PRETRAINED_FIGURES_RUN.json).
All predictor weights load with no missing, unused, or mismatched keys.
SmolLM2 has 134,515,008 loaded parameters and GPT-2 has 124,439,808.

The fixed text is `The red car stopped near the bridge.` Both tokenizers
produce eight positions, with model-specific IDs. Each model replays its
own IDs without sampling or training. All weights are frozen, evaluation
mode disables dropout, and CPU float32 inference uses one thread,
deterministic algorithms, and native eager attention. SmolLM2's checkpoint
stores BF16 weights but its inspected cache is float32.

Choices are fixed before inspection: prefill length two; probe `car` at
array index two; block one; first KV group and its three query heads;
first eight displayed features. Token positions and feature indices start
at zero, while printed head labels start at one. A middle dot replaces a
tokenizer boundary marker for display; it is not an extra input character.
The probability bars select the top five IDs by a fixed rank rule under
the correct full pass, rather than searching for attractive completions.

## Four figures

| File | Observations |
| --- | --- |
| `ch10_pretrained_cache.png` | Native rotated keys and values after two, three, and eight consumed inputs; old rows preserved exactly |
| `ch10_pretrained_shared_reads.png` | Three native attention distributions reading the same KV head; signed weighted-value contributions |
| `ch10_pretrained_replay.png` | Full-versus-cached vocabulary-logit differences in both models; probabilities after a deliberate RoPE-position reset |
| `ch10_pretrained_bytes.png` | Actual tensor payload across all layers at every stored length, checked against shape arithmetic |

Gray cache cells represent absent future positions, not zero-valued vectors
or padding. All three key panels use one signed scale; value panels use
their own common signed scale. The probability heatmap uses a common scale
from zero to one. Figure excerpts do not replace full-array checks.

## Complete replay checks

The script compares a full eight-position causal pass against schedules
`(2,1,1,1,1,1,1)` and `(2,1,2,3)`. Every call passes the correct absolute
`position_ids`, `cache_position`, and complete prefix attention mask.
Both schedules consume the same IDs. Every vocabulary logit is checked
against the full pass with absolute tolerance `5e-4` and zero relative
tolerance. The recorded maxima across both schedules are about `6.87e-5`
for SmolLM2 and `3.21e-4` for GPT-2. Every highest-logit token ID agrees.
All future attention entries in the full pass are exactly zero.

`DynamicCache` mutates as inputs are appended, so snapshots copy arrays
before the next call. The old slice is checked for exact equality after
every append in both models and schedules. This is different from matching
a fresh full pass, where matrix multiplication groupings can produce small
float32 differences. Final cache arrays from both schedules are compared
with the full-pass cache, across all layers and coordinates. The largest
single-step cache discrepancies are about `9.35e-5` for SmolLM2 and
`2.39e-5` for GPT-2.

At the fixed probe in SmolLM2, forward hooks capture the normalized first
attention input, exact cosine/sine arrays, and concatenated head outputs
before the output projection. Independent float64 query projections use
all 576 input coordinates. Applying the captured split-half RoPE and
reading the actual cached keys reconstructs all nine attention rows within
`1.93e-7`. Their weighted sums over complete 64-coordinate cached values
match the native head outputs within `4.20e-8` in the recorded run.

The first query head's scores are approximately `(2.275314, 1.516421,
3.045020)` and its weights are `(0.275686, 0.129072, 0.595242)`.
Heads two and three share exactly the same stored KV head yet give
`(0.249591, 0.330867, 0.419542)` and `(0.304370, 0.672940, 0.022690)`.
The first value coordinates at the three positions are approximately
`(-0.024864, 0.013622, -0.051439)`. Weighted contributions
`(-0.006855, 0.001758, -0.030619)` sum to head coordinate `-0.035715`.
Probabilities are nonnegative, while the values and output can be signed.

## Controlled position error

After a fresh correct two-token prefill, the script consumes position two
with the correct cache slot and mask but forces only its RoPE
`position_ids` to zero. The weights, input IDs, old cache rows, and key
permissions are retained. Maximum vocabulary-logit change at that call is
about 8.36127. The probability of ` was` changes from about 0.096066 to
0.077142. The five displayed correct-pass IDs contain about 24.77% of its
probability mass; omitted vocabulary entries retain the rest. This is a
local bug demonstration, not a sampled continuation or a model-quality
evaluation.

## Payload accounting

Every returned K/V tensor contributes `numel() * element_size()` bytes.
The shapes retain three KV heads in SmolLM2 and twelve in GPT-2, even
though the eager attention implementation can expand/broadcast histories
temporarily to query heads. Every length is checked against
`batch * layers * length * kv_heads * (key_width + value_width) * 4`.

At batch size one, SmolLM2 adds 46,080 bytes (45 KiB) per position and
stores 368,640 bytes (360 KiB) at eight positions. GPT-2 adds 73,728 bytes
(72 KiB) per position and stores 589,824 bytes (576 KiB) at eight.
These are real float32 payloads, not hypothetical 16-bit counts. They
exclude weights, allocator reservations, working tensors, and Python
metadata. The checkpoints differ in dimensions and weights, so the curves
are not a controlled speed or quality comparison of MHA and GQA.

The [Transformers 4.57.1 Llama implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py),
[GPT-2 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gpt2/modeling_gpt2.py),
and [cache implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py)
define the inspected execution. The [GQA paper](https://arxiv.org/abs/2305.13245)
describes the head-sharing architecture. The chapter's MLA section remains
a separate mathematical explanation and small NumPy example; these figures
do not claim to execute a DeepSeek model or compressed latent cache.

## Reproduction record

The compact JSON record contains the November 9, 2026 observations,
checkpoint and source hashes, exact IDs, dimensions, software, numerical
checks, stage sizes, and plot/archive hashes. A fresh run also writes
`observations.npz` with complete logits, all-layer final caches, copied
single-append cache stages, and full probe attention internals. Generated
arrays and images remain local; only source and compact metadata are
versioned in the public repository. Archive and image hashes can vary
across library versions, so inspect numerical agreement when reproducing.
