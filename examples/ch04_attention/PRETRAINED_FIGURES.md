# Open one real self-attention head

The chapter's three optional figures inspect the first block and first head
of [`distilbert/distilgpt2`](https://huggingface.co/distilbert/distilgpt2/tree/2290a62682d06624634c1f46a6ad5be0f47f38aa),
at revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`.
These locations and the first eight displayed coordinates are chosen in advance.
The model card labels this checkpoint Apache 2.0. We use built-in Transformers
code and safetensors weights, without executing remote Python.

The frozen model runs in evaluation mode on one CPU thread, using float32
and eager attention. There is no training, dropout, sampling, or generated
text. The regular NumPy lesson and its tests need none of these dependencies.

From the repository root:

```sh
python -m pip install -r requirements-appendix-figures.txt
python -m examples.ch04_attention.pretrained_figures \
  --cache-dir /path/to/huggingface/hub \
  --output-dir output/ch04-pretrained
```

The initial run downloads the pinned public checkpoint. Add `--offline` to
reuse cached files. `PRETRAINED_FIGURES_RUN.json` next to this guide records
the reference inputs, IDs, results, package versions, and source, weight,
array, and figure checksums. Different platforms can change floating-point
results; the checks use explicit tolerances rather than exact reference bytes.

## What is captured

The checkpoint has six blocks, 12 heads per block, model width 768, and
head width 64. Inputs are `the red car` and `the red bike`, tokenized without
extra special tokens. Both are three tokens long. Their first two IDs match,
and their final IDs differ. A middle dot in the plots denotes a real space
in a decoded token piece.

Hooks observe the first attention block's projection input, joint Q/K/V
projection output, and concatenated head reads entering the output projection.
The model also returns attention weights and hidden states. Hooks are removed
after the two passes, including on failure. The layer-normalized projection
input already includes position information. GPT-2 uses learned affine
projections with biases. Its `Conv1D` projection stores a matrix that is used
as `input @ weight + bias`; it does not convolve across token positions.
The observations are captured before head combination's output projection,
residual addition, and later layers. They are not final vocabulary scores.

The reference weight-file SHA-256 is
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.

## Three measured views

`ch04_pretrained_qkv.png` displays the first eight of each head's 64 Q/K/V
coordinates for `the red car`, with one shared signed color scale.
The query, key, and value retain the same input-position identities, but
their learned projections differ. Features have no assigned semantic names,
and their magnitudes are not probabilities or independent importance scores.

`ch04_pretrained_read.png` shows finite scaled scores before masking, the
native causal weights, and two coordinates of the value contributions for
the final ` car` query. All 64 coordinates enter each dot product, with
divisor `sqrt(64) = 8`. Reference measurements for that row are

```text
raw dot products: (13.605484, 0.583237, -2.399441)
scaled scores:   ( 1.700685, 0.072905, -0.299930)
source weights:  ( 0.750967, 0.147463,  0.101570)
read features 1 and 2: (-0.126077, -0.078598)
```

The ` red` query has scores approximately `(1.913109, 0.142096, 1.463292)`.
The last score is substantial but its future key receives exactly zero weight.
The first query has only one allowed key and must give it weight one.
Those facts distinguish the mask's information rule from learned preferences.
The ` car` row puts most weight on `the`; one head is not a unique
grammatical explanation. Rounded display values never replace the raw arrays.

`ch04_pretrained_causality.png` changes only the last input token to ` bike`.
The two attention panels share a 0–1 scale; the read-difference panel uses a
separate signed scale. The first two head reads remain identical across all
64 coordinates. The maximum absolute change at the last position is
approximately `0.050881`. All returned prefix hidden states throughout the
model also agree exactly in this recorded run (checked with tolerance `1e-6`).
This is a controlled future-input change, not a test of language quality.
The changed token alters its Q, K, and V together; this differs from changing
only V while holding the attention weights fixed.

## Numerical verification

The inspection reconstructs the affine projection in float64 from captured
inputs and learned parameters. It then passes the observed float32 Q/K/V
through this chapter's independent float64 NumPy attention implementation.
All 12 heads are checked against the native weights and native head reads.
The reference maximum differences in those attention-core checks are about
`1.7e-7` and `1.2e-7`, respectively. The tolerances allow floating-point
differences, not a change to the mathematical operation.

Other checks require normalized rows, exactly zero future weights, the
explicit 64-product dot sum, value contributions agreeing with the native
read, and the convex bounds of each read coordinate. The prefix control
compares projected inputs, Q/K/V, head reads, and all returned hidden layers.

The output directory contains the three PNGs, a run manifest, and
`ch04_pretrained_observations.npz`. The archive saves the complete projected
inputs, Q/K/V for all heads in this block, native and reconstructed reads
and weights, full hidden states, scores, and the selected contributions.

```python
import numpy as np

a = np.load("output/ch04-pretrained/ch04_pretrained_observations.npz")
print(a.files)
print(a["original_q"].shape)  # (12 heads, 3 positions, 64 features)
print(a["original_attention"][0])
np.testing.assert_allclose(a["last_query_contributions"].sum(0),
                           a["original_head_reads"][0, -1], atol=2e-5)
```

The core lesson remains separately executable and checked:

```sh
python -m examples.ch04_attention.run_examples
python -m unittest discover -s tests -p test_attention.py -v
python scripts/export_snippets.py --check
```

The pinned implementation can be inspected in the
[Transformers v4.57.1 GPT-2 source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gpt2/modeling_gpt2.py).
The mathematical attention operation follows
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).
