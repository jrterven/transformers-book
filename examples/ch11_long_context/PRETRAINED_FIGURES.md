# Real Longformer windows, global relays, and retained mass

The optional experiment inspects the published
[allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096)
checkpoint at revision `301e6a42cb0d9976a6d6a26a079fef81c18aa895`.
The model card labels the weights Apache 2.0. The ordinary NumPy examples
do not require these dependencies or a model download.

From the repository root:

```sh
python -m pip install -r requirements-ch11-figures.txt
python -m examples.ch11_long_context.pretrained_figures \
  --cache-dir /path/to/huggingface/hub \
  --output-dir output/ch11-pretrained
```

Add `--offline` after populating the cache. The official checkpoint stores
about 597 MB in `pytorch_model.bin` and has no safetensors file at this
revision. The loader explicitly uses `weights_only=True`, PyTorch 2.6 or
later, `token=False`, and `trust_remote_code=False`. No repository Python
code is executed and the input text is never uploaded. Model weights and
generated arrays are not redistributed here.

## Fixed input and probes

The input begins `The committee first met in Paris. `, repeats
`The report records routine dates, names, budgets, and project updates. `
45 times, and ends `In which city did the committee first meet?`.
Including start/end markers, the fast tokenizer produces 648 positions.
The fixed source rule chooses the first `ĠParis` piece at index 6;
the relay is `floor(648/2) = 324`, a piece of `budgets`;
the target is the last `Ġcity` piece, index 640. The displayed head is
head zero of block zero. These choices precede model inspection.
Printed head/block labels start at one; array positions start at zero.

The checkpoint has twelve blocks, twelve heads, 768 hidden features,
64 features per head, and learned absolute positions. Its native
`attention_window=512` gives radius 256, including up to 513 local keys
with the query's own position. The implementation pads to 1,024 positions
for its kernel. Padding keys have zero probability and are omitted from
the plots. Published window settings and weights are unchanged.

The model is loaded as `LongformerForMaskedLM`, and observations call its
complete `longformer` encoder without allocating unused vocabulary logits.
All predictor weights load with no missing or mismatched tensors. The file
also contains exactly two unused pooler tensors (`dense.weight` and
`dense.bias`), which this masked-language-model class does not instantiate
or execute; these are recorded explicitly in the run JSON. No newly
initialized weights contribute to the observations. There are 148,711,257
loaded parameters. Inference uses CPU float32, evaluation mode, frozen
weights, one thread, and deterministic algorithms, with no training or
sampling.

## Four figures and what they measure

| File | Actual observations |
| --- | --- |
| `ch11_pretrained_windows.png` | Full first-head probability maps with local-only and one-global masks |
| `ch11_pretrained_relay_reads.png` | Ordinary question-query and global-query probability rows |
| `ch11_pretrained_perturbation.png` | Complete 768-coordinate sensitivity to one Paris-to-Rome ID replacement |
| `ch11_pretrained_retained_mass.png` | Post-hoc retained mass, complete 64-coordinate head-output error, and its mathematical bound |

The first two figures expand native compact attention outputs solely for
inspection. They are not evidence that native execution allocates dense
score matrices. Local-only connectivity has 266,632 allowed pairs. Adding
the midpoint's global row and column yields 266,902. All disallowed weights
are exactly zero; all allowed entries have positive weights in this run.
The probability maps share a logarithmic scale; gray cells are exact zeros.
The row plots use a small linear region around zero and logarithmic spacing
above it, preserving exact zero weights on the baseline.

Ordinary queries read global keys and local keys in one softmax, using the
ordinary learned projections. Global keys are excluded from local slots to
avoid double counting. Global queries use separate learned global Q/K/V
projections. The local attention output's all-zero global-query row is a
filler; its real weights are returned separately as `global_attentions`.
The script uses that separate array, rather than mistaking the filler for
an absence of global attention.

Forward hooks capture the first attention input and concatenated head
outputs. Independent float64 projection and softmax calculations check
every probability in the displayed head under both masks, with largest
absolute error below `1.25e-6`. The complete 64-coordinate head outputs
agree with native execution within `1.85e-6`. Native float32 row sums differ
from one by at most `4.37e-7`. Checks include global-query outputs with
their separate value projection and exactly zero padding weights.

## Single-token perturbation

Four complete encoder passes use the two masks on the base and changed
texts. The changed input replaces only the ID at source 6 by the single ID
for ` Rome`; all other IDs, positions, masks, and weights are fixed.
Sensitivity is the maximum absolute difference across all 768 features
at embedding output, block 1, and block 2. Comparisons are always between
two texts with the same mask.

The source-to-target distance is 634, greater than the two-local-layer
reach of 512. The target's first two local-only block outputs are exactly
unchanged. With a global midpoint, that midpoint changes by about
`0.00081238` after block 1, and the question changes by about `4.76837e-6`
after block 2. The question is unchanged after block 1. The complete blocks
include all twelve heads; the row plots show only the first head. In that
head the question's first-block weight on the global relay is about
`6.69814e-6`, and the relay's weight on the source about `1.52304e-5`.
The small response illustrates a measured information path. It is not a
factual-answer evaluation or a causal-generation experiment.

## Selection diagnostic and bound

The target's first-head global-enabled row has 265 legal keys: 264 local
positions plus global key 324. Rank its already-computed native weights,
retain `k = 1,2,4,8,16,32,64,128,256,265`, renormalize, and mix the actual
ordinary projected values. The float32 weights are normalized once in
float64 to remove their row-sum rounding. Independent gathered-softmax
checks use the actual projected main scores.

If the retained mass is `rho`, the retained conditional output is `oS`,
and the discarded conditional output is `oR`, then
`o = rho*oS + (1-rho)*oR`. Hence
`||oS-o|| <= 2*Vmax*(1-rho)`, where `Vmax` is the largest norm of any legal
value. The script checks this inequality at every displayed count.
For `k=1`, retained mass is about 0.55255 and output error 1.65226.
For `k=8`, mass is about 0.92339 and error 0.16536. All 265 entries restore
the reference up to float64 summation rounding (`6.62e-16`). The measured
`Vmax` is about 3.39580. This is a local head-output diagnostic using all
scores first, not an efficient indexer, DSA execution, or a model speedup.

The [Longformer paper](https://arxiv.org/abs/2004.05150) introduces local
and global attention. The
[pinned Transformers implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/longformer/modeling_longformer.py)
defines the inspected arrays and projection families. The chapter's DSA,
CSA, and HCA explanations remain separate theory and small NumPy examples;
these figures do not claim to run a DeepSeek model.

## Reproduction record

The [compact run record](PRETRAINED_FIGURES_RUN.json) contains checkpoint
file SHA-256 hashes, exact input IDs and pieces, software, probe rules,
numerical checks, sensitivity measurements, selection curves, and output
hashes. A fresh run also writes `observations.npz` with native compact
arrays, expanded probabilities and masks, actual projected vectors,
complete first-two-block hidden states, and the diagnostic curves.
Only source and compact metadata are versioned in this public repository.
Numerical agreement is more informative than byte-identical plot/archive
hashes across library versions.
