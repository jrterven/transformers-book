# Selective memory in real Mamba and Mamba-2 models

The six optional figures use two published checkpoints without additional
training. The ordinary NumPy examples and ten exercises remain offline.

```sh
python -m pip install -r requirements-ch14-figures.txt
python -m examples.ch14_selective_ssm.pretrained_figures \
  --cache-dir output/model-cache --source-dir output/ch14-source \
  --output-dir output/ch14-pretrained
```

Add `--offline` after the cache has been populated. The program saves
`observations.npz`, six PNGs, and `run.json`; the checked record is also in
[PRETRAINED_FIGURES_RUN.json](PRETRAINED_FIGURES_RUN.json). The run uses one
CPU thread, float32 model evaluation, deterministic PyTorch algorithms,
evaluation mode, frozen weights, and float64 independent reconstruction.
It does not sample text or measure device throughput.

## Checkpoints and CPU references

- [state-spaces/mamba-130m-hf](https://huggingface.co/state-spaces/mamba-130m-hf/tree/1e76775f628fbf1350fbe4dbb3d971ba64af25a1),
  revision `1e76775f628fbf1350fbe4dbb3d971ba64af25a1`. The safetensors file
  is about 517 MB; the model has 129,135,360 unique parameters. It loads
  through Transformers 4.57.1 without missing, unexpected, or mismatched
  keys. Its model card labels the weights Apache 2.0.
- [state-spaces/mamba2-130m](https://huggingface.co/state-spaces/mamba2-130m/tree/3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49),
  revision `3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49`. The original PyTorch
  checkpoint is about 258 MB and has 128,989,632 unique parameters. It is
  loaded with `weights_only=True`, then evaluated in float32. The tiny
  release card does not provide a weights license; consult its release terms
  for reuse. No weights are redistributed here.

The original Mamba-2 repository uses a different configuration schema from
Transformers. The script explicitly supplies the published/default dimensions:
24 layers, width 768, expanded width 1536, state width 128, 24 heads of
width 64, one write/read group, short filter width 4, and padded vocabulary
50288. The installed model's load hook renames `embedding` to `embeddings`.
Every weight loads with `strict=True`. The tied predictor matrix is checked
against the released tensor. No parameters are left at random initialization.
The one-group gated RMSNorm and per-head direct path match the original
block defaults. This is an explicit checkpoint port into a CPU reference,
not a claim that the original repository's configuration can be passed
unchanged to an automatic HF model loader.

We also verify the state-space calculation against the authors'
[minimal SSD implementation](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/ssd_minimal.py),
commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`. Its SHA-256 is checked
before use. Only the unchanged bodies of `segsum` and
`ssd_minimal_discrete` are compiled from the reviewed source. CUDA imports
and the unrelated CUDA comparison test are omitted. The allowlisted source
hash is in the script and run record; future revisions require new review.

The installed library's Mamba and Mamba-2 CPU source hashes are recorded.
We do not install or benchmark fused CUDA kernels. The model loaders use
`token=False` and `trust_remote_code=False`; public source downloads contain
no user inputs. Inputs, model weights, and downloaded implementation files
are not uploaded by the experiment.

## Fixed probe

Tokenize the following string repeated 40 times using the pinned Mamba HF
tokenizer, without special tokens:

```text
The parcel arrived in Paris. The weather was cloudy. The parcel moved to Rome. The driver checked the map. 
```

Retain the first 386 IDs. The first 384 form the prompt; the next two IDs
are consumed in separate cached calls. The tokenizer uses the GPT-NeoX
vocabulary shared by the original Mamba releases. The exact IDs and tokenizer
pieces are saved in the run record. The natural-language text is a controlled
replay, not a delivery-location accuracy benchmark.

Before observing outputs, fix block zero, Mamba channels 0--2, Mamba-2 head
zero, the read at array position 95, and 32-token native CPU chunks. Printed
positions, channel names, and head names start at one. A separate intervention
replaces ID 191 (a period in this probe) with the single token for ` blue`.
Every other ID and the length stay fixed. The replacement is made in the
token array rather than by retokenizing a changed text string.

## Observed algebra

Mamba hooks capture the native first mixer input and its output before the
last projection. Reconstruct the short causal convolution, activation, input
projection, positive steps, learned negative rates, and time-varying B/C.
For channel zero, a float64 recurrence uses the actual discrete write
`delta * B * u`. Its final state agrees with the native cache within
`1.58e-7`. Including the real direct D path and SiLU output gate reproduces
the native pre-projection channel within `3.65e-8`. Both assertions use an
absolute tolerance of `3e-5` and zero relative tolerance.

The channel coefficient matrix sums the 16 state-coordinate paths. It maps
the continuous SSM input features to the SSM core output with controls held
fixed, excluding D and the output gate. Matrix multiplication agrees with
the recurrence within `1e-11`. It is neither an attention probability matrix
nor a derivative through all of the input-derived control branches.

For Mamba-2, hooks capture the native features and output before its gated
normalization. The official minimal SSD function receives every head and
feature, with `X = delta * u`, log retention `delta * A`, and the actual B/C.
It agrees with the library reference, including the D path, within
`3.052e-5` over 384 positions, 24 heads, and 64 features. The final head
states agree within `1.91e-6`. Both assertions allow `2e-4` absolute error.

Independently, the NumPy head uses raw `u`, effective write `b = delta * B`,
and retention `exp(delta * A)`, applying the step exactly once. In head
zero, recurrent, dense, and chunked outputs agree within `1e-10`, including
chunks of length 37 with a shorter final chunk. This head agrees with the
native core within `2.77e-6`, against a `2e-4` absolute tolerance. The book's
state convention is `[N, P]`; the native cache convention is `[P, N]`.

## Continued inference and causality

At prefix lengths 64, 128, 384, 385, and 386, measure actual cache tensor
payloads for all 24 layers. Batch size is one and every cache value is
float32. Each model has the same payload at all five lengths:

| Model | SSM bytes | Convolution bytes | Total MiB |
| --- | ---: | ---: | ---: |
| Mamba 130M | 2,359,296 | 589,824 | 2.8125 |
| Mamba-2 130M | 18,874,368 | 688,128 | 18.65625 |

Mamba's per-layer shapes are `[1, 1536, 16]` for state and
`[1, 1536, 4]` for the convolution buffer. Mamba-2 uses
`[1, 24, 64, 128]` and `[1, 1792, 4]`. The expanded short-convolution
branch includes B and C in the latter model. These are array payloads,
excluding model weights, allocator overhead, temporary allocations, saved
logits, and training activations. They compare two different configurations,
not intrinsic memory requirements for all models of either family.

The two cached calls are compared with a full-prefix replay, over every
output-vocabulary coordinate. Maximum logit discrepancies are `2.709e-4`
for Mamba and `2.900e-4` for Mamba-2; both two-token argmax comparisons
agree. Assertions allow `1e-3` absolute discrepancy with zero relative
tolerance. They check numerical equivalence, not generated-text quality.

The token-192 intervention leaves all earlier logits exactly unchanged in
both recorded runs. Current and subsequent logits change. The plot takes
the maximum absolute difference across all 50280 / 50288 output coordinates,
including padded entries. Unnormalized score differences can be large and
do not directly measure probability change or task relevance.
The earlier-prefix assertion permits `1e-4` for reproducibility across
numerical environments, and the suffix must respond by more than `1e-3`.

## Figures

| File | Evidence |
| --- | --- |
| `ch14_pretrained_mamba_controls.png` | Captured steps and reconstructed coordinate retentions |
| `ch14_pretrained_mamba_paths.png` | Frozen-control source coefficients and contributions to read 96 |
| `ch14_pretrained_ssd_matrices.png` | Actual head's decay, content, and effective sequence matrices |
| `ch14_pretrained_ssd_chunks.png` | Local plus incoming-state output and state magnitudes at boundaries |
| `ch14_pretrained_cache_equivalence.png` | Persistent cache payloads and cached/full-prefix logit agreement |
| `ch14_pretrained_causality.png` | Complete-model intervention with an explicit earlier-prefix panel |

Signed coefficient images use a symmetric logarithmic scale with a linear
region near zero (`linthresh=1e-3`). Blocked future cells are gray; exact
allowed zeros remain at the color scale's center. The SSD content and
effective matrices share the same color limits. The boundary-state heatmap
uses a logarithmic RMS-magnitude scale across the 64 feature columns, without
normalizing state-coordinate rows.
