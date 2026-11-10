# Real attention arrays and equivalent execution

The optional experiment uses
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The model card labels the
weights Apache 2.0. The ordinary chapter examples remain NumPy-only.

From the repository root:

```sh
python -m pip install -r requirements-ch12-figures.txt
python -m examples.ch12_efficient.pretrained_figures \
  --cache-dir /path/to/huggingface/hub \
  --output-dir output/ch12-pretrained
```

Add `--offline` once the cache is populated. The first run downloads a pinned
public safetensors checkpoint (about 270 MB) and its tokenizer. The loader
uses `token=False`, `trust_remote_code=False`, and `use_safetensors=True`.
No custom repository Python is executed and no input text is uploaded.
Model weights and generated arrays are not redistributed here.

## Fixed experiment

SmolLM2 has 30 blocks, hidden width 576, nine query heads, three KV heads,
and head width 64. All 134,515,008 predictor parameters load without missing,
unused, or mismatched keys. The first three query heads read KV head zero.
The experiment fixes block zero, query head zero, probe query 383, key tiles
of width 64, and feature coordinate zero before inspecting any outputs.
Printed block/head labels start at one; all array positions start at zero.

Tokenize the string
`The red car stopped near the bridge. The driver checked the map before continuing. `
repeated 40 times, without added special tokens. Retain its first 514 token
IDs. The first 512 are prefill; the following two IDs are consumed in two
one-token cached calls. Exact IDs and tokenizer pieces are in the run
record. This is a controlled replay, not a sampled continuation or a
natural-document benchmark.

All weights stay frozen, evaluation mode disables dropout, CPU float32
uses one thread and deterministic algorithms, and no training occurs.
The runtime is PyTorch 2.15.0 with Transformers 4.57.1. The recorded host is
macOS on ARM64; the restricted environment reports the processor family
as `arm` rather than a chip model. Timing observations describe this local
setup and do not establish performance on other CPUs or GPUs.

## Four figures

| File | Evidence |
| --- | --- |
| `ch12_pretrained_stage_reads.png` | Native first-head probabilities: 512-by-512 prefill and 1-by-513 first cached call |
| `ch12_pretrained_online_state.png` | Maximum, denominator, first numerator coordinate, and output ratio for actual query 383 |
| `ch12_pretrained_allocations.png` | Measured positive self-CPU-memory events for the native first attention module |
| `ch12_pretrained_equivalence_timing.png` | Complete-model vocabulary-logit agreement and separately measured component timings |

Both attention maps use the same logarithmic scale; gray cells are exact
zeros for forbidden keys. Display guides mark 64-by-64 teaching tiles.
Their shapes are independent of the internal tiling selected by the CPU
kernel. The orange row identifies the fixed online-softmax probe.

## Native arrays and online recurrence

Forward hooks capture the normalized first attention input, native RoPE
cosine/sine arrays, first-head probabilities, and concatenated head outputs
before the output projection. Query reconstruction uses all 576 input
coordinates and the actual learned query matrix. The native prefill cache
supplies rotated keys and values from the shared KV head. Independent
float64 reconstruction checks the complete first-head probability matrix
and all 512-by-64 head outputs, including exact causal zeros.

In the recorded run, the largest probability discrepancy is about
`2.06137e-7` and the largest head-output discrepancy `7.75423e-8`.
The chapter's tiled function, with uneven query tiles of 37 and key tiles
of 64, agrees with the float64 materialized reference within `9.71446e-17`.

For query 383, six key tiles cover all 384 permitted keys. The last two
tiles contain only future positions, and the script checks exact equality
of every state array before and after those masked updates. After tile six,
`m = 4.7203023882`, `ell = 33.3609136537`, and
`u[0] = -0.1020124165`, yielding `o[0] = -0.0030578424`.
The complete 64-coordinate result agrees with native execution within
`2.54551e-8`. These are teaching recurrence states computed from captured
model vectors, not instrumented private accumulators inside the CPU kernel.
The diagnostic reconstructs a dense score array for inspection; the
separate tiled evaluator skips wholly forbidden tiles.

## Complete-model equivalence and backend identification

The same loaded model switches between Transformers' `eager` and `sdpa`
attention implementations. Each complete pass uses identical token IDs,
position IDs, cache positions, and legal causal pairs. Eager represents
causality with an additive mask; SDPA prefill can use its causal flag.
Requesting attention maps from SDPA can change supported execution, so
the SDPA run uses `output_attentions=False`; its backend is identified
from the profiler trace instead.

Every vocabulary logit at all 514 consumed positions is compared, not
just the highest-probability token. Maximum absolute discrepancy is
`7.82013e-5` and every highest-logit ID agrees. Complete final cache arrays
across all 30 layers agree within `6.29426e-5`. The tests allow `5e-4`
absolute logit error and `2e-4` absolute cache error, with zero relative
tolerance. These are floating-point equivalence checks with fixed inputs,
not generated-text quality measurements.

The observed SDPA path includes
`aten::_scaled_dot_product_flash_attention_for_cpu` and
`aten::scaled_dot_product_attention`. The script requires the CPU fused
operator to appear in its profiled calls rather than labeling an unknown
backend as fused. This CPU implementation is separate from CUDA
FlashAttention-1 through -4 and their GPU benchmark results.

The native first-module outputs are also checked under both backends for
each profiled prefix. Prefixes use captured embedding-stage normalized
inputs and their original positional rotations. Since embedding and input
normalization are positionwise, these are valid first-block prefix inputs.
The module includes Q/K/V projections, positional rotations, attention,
and the output projection. It excludes the feed-forward sublayer, later
blocks, KV-cache allocation, and vocabulary projection.

## Allocation measurements

Each profiled first-module call uses actual prefixes of length 64, 128,
256, or 512. The profiler records operator input shapes and CPU memory.
Figure (a) takes the largest positive `self_cpu_memory_usage` value.
Figure (b) sums positive values of this exclusive statistic. Exclusive
values avoid adding a parent's nested child allocations again. These are
positive operator-memory event summaries, not peak live memory or a count
of bytes read/written. Preexisting inputs, masks, and weights are outside
the measured call; deallocations are excluded from the positive sum.

At 512 positions, eager's score multiplication, masked-score addition, and
softmax each have a 9 MiB positive event. The softmax input shape is
`[1,9,512,512]`, and its output occupies exactly `9*512*512*4` bytes.
SDPA's largest positive event in this module is 1.125 MiB. The positive
event totals at 512 are about 50.25 MiB for eager and 11.03 MiB for SDPA.
Those totals include many temporary allocations and can exceed the amount
live at any one time. Neither statistic measures GPU HBM traffic.

The compact record includes the eight largest positive events with input
shapes, all separate softmax events, aggregate statistics, and actual SDPA
operator names for each call. It does not claim to be a full allocator
timeline or a hardware-counter trace.

## Timing protocol

Timing is separate from profiling. Each implementation receives three
warmups per prefix length and eleven timed synchronous CPU calls, using
`perf_counter_ns`. Implementation order alternates across samples.
Backend switching and output cleanup occur outside timed intervals.
Points are medians; whiskers span the 25th through 75th percentiles.
Every raw duration is saved in milliseconds. These component measurements
exclude the rest of the model and serving overhead. They are not response
latencies, throughput estimates, confidence intervals, or GPU speedups.

The [FlashAttention paper](https://arxiv.org/abs/2205.14135) introduces the
IO-aware GPU schedule. The
[pinned Llama implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py)
and [SDPA integration](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/integrations/sdpa_attention.py)
define the inspected model paths. The
[PyTorch SDPA documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)
describes interface dispatch; the trace establishes this run's backend.

## Reproduction record

[The run record](PRETRAINED_FIGURES_RUN.json) contains checkpoint file
SHA-256 hashes, exact IDs and pieces, dimensions, native mask shapes,
operator observations, numeric checks, all timing samples, software,
source hash, and output hashes. A fresh run also writes `observations.npz`
with both complete logit arrays, all-layer final caches, native first-head
probabilities and values, captured/reconstructed head outputs, and state
trajectories. Only source and compact metadata are versioned publicly.
Timing and plot/archive hashes can vary; compare numerical agreement and
actual operator names when reproducing.
