# Six observations from a released small hybrid

Run from the companion repository root:

```sh
python -m pip install -r requirements-ch15-figures.txt
python -m examples.ch15_hybrids.pretrained_figures \
  --cache-dir output/model-cache --output-dir output/ch15-pretrained
```

Add `--offline` after the first download. To redraw the saved observations
without loading the model, add `--render-only`. The program writes six PNGs,
`observations.npz`, and `run.json`. The [recorded run](PRETRAINED_FIGURES_RUN.json)
contains the exact versions, source/checkpoint hashes, token IDs, head choice,
cache shapes, numerical checks, and image hashes.

## Source and scope

- Model: [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68),
  revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, Apache-2.0.
- Runtime: the built-in [Transformers 5.13.0 CPU reference](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
  PyTorch 2.15.0, float32, four CPU threads, evaluation mode, deterministic
  algorithms. No custom model code, Hub kernels, extra training, or sampling.
- The native conditional-generation class loads the published multimodal
  checkpoint. This experiment feeds text only and executes the complete
  24-block language pathway and vocabulary projection. It supplies no vision
  input and does not invoke optional multi-token prediction branches.
- Its language configuration has 18 GDN and six full-attention blocks in
  repeated 3:1 groups, hidden width 1024, 16 GDN heads of key/value width 128,
  and eight attention query heads sharing two KV heads of width 256.
  Feed-forward sublayers are dense. It is a different model from Jamba and
  Olmo Hybrid; those models are not executed by this program.

## Fixed probe and selected features

The supplied text repeats a parcel narrative; its first 128 token IDs form the
prefill, and the next two are replayed as cached inputs. Tokenization does not
add special tokens or a chat template. All other sequence details are recorded.

The controls figure shows the first three heads of GDN block 1. Detailed writes,
frozen-control coefficients, and chunk contributions use the second displayed
head, whose near-one retentions contrast with the first head's frequent strong
forgetting. They show input position 48 and the first eight value coordinates
for the write, the first 48 positions for the coefficient matrix, and feature 1
over 96 positions for chunk curves. This explicit illustrative selection is not
a search for a better accuracy score. Full-attention weights come from block 4,
query head 1. Indices printed in the chapter start at one; constants in code and
the run record start at zero.

The native key/query normalization divides by the square root of the squared
norm plus `1e-6`; native queries are additionally scaled by `1/sqrt(128)`.
Captured vectors are normalized in native float32 before independent NumPy
float64 calculations. The native state is `[batch, head, key, value]`, the
transpose of the book's `[value, key]` matrix. Alpha is `exp(g)` and beta is a
sigmoid in `(0,1)`; this is not Olmo Hybrid's doubled-sigmoid extension.

## Numerical validation

The program checks all of the following before saving a successful run:

1. No required weights are missing or mismatched. Any reported unused weights
   must belong to an optional MTP branch. The recorded native loader reports
   no missing, unexpected, or mismatched keys.
2. Native sequential and chunked GDN cores, all 16 heads and all value features,
   including their final states, agree within absolute `2e-4`.
3. A separately implemented NumPy recurrence reconstructs the selected head and
   transposed final state within `2e-4`; its triangular chunk solver with chunk
   length 31 agrees with its recurrence within `1e-11`, including the short
   last chunk. The source code never substitutes a scalar incoming read for
   the full incoming-state response through subsequent corrections.
4. Frozen-control coefficient probes reconstruct all selected-head outputs
   within `1e-11` when multiplied by the original captured value vectors.
   Norm-aware key-read interpolation is also checked over all coordinates.
5. Native attention weights are causal and normalized within `2e-6`.
   Multiplying those weights by the actual cached values, then applying the
   native output gate, reconstructs the input to the output projection within
   `2e-5`, across all attention heads and features in the observed block.
6. Cached tokens 129 and 130 agree with full-prefix vocabulary logits within
   `2e-3`, and both argmax decisions agree. The observed maximum discrepancy
   is below `4e-5`; the acceptance tolerance is distinct from that measurement.
7. Replacing only token 64 with the single token for ` blue` changes suffix
   logits, while the earlier prefix changes by less than `1e-4` (exact zero
   in the recorded run). All 248,320 output coordinates enter the maxima.
8. Actual cache payloads at 32, 64, 128, 129, and 130 tokens match the named
   arrays. GDN matrix states total 18 MiB, convolution buffers total 1.6875 MiB,
   and KV histories grow by 24 KiB per token. Persistent history therefore
   occupies `19.6875 + 0.0234375*n` MiB in this float32, batch-one replay.

The observations measure internal computations and payloads. They do not assess
recall accuracy, semantic importance, architecture quality, optimized CUDA
performance, or peak process memory. GDN coefficients are signed frozen-control
linear maps, not attention probabilities or the full input-output Jacobian.
The state/count figures exclude weights, allocator overhead, saved output
traces, and temporary computation. Different precisions change the byte counts.
