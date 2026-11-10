# Learned S4 memory and HyenaDNA gates

The six optional figures inspect two released models, with no additional
training. The ordinary examples and eleven exercises still use only NumPy.

```sh
python -m pip install -r requirements-ch13-figures.txt
python -m examples.ch13_state_spaces.pretrained_figures \
  --cache-dir output/model-cache --source-dir output/ch13-source \
  --output-dir output/ch13-pretrained
```

Add `--offline` after the first run. The program writes the measured arrays
to `observations.npz` and a provenance/check record to `run.json`. The checked
run is also recorded in [PRETRAINED_FIGURES_RUN.json](PRETRAINED_FIGURES_RUN.json).
It uses CPU float32, one thread, frozen weights, evaluation mode, a fixed
seed, and deterministic PyTorch algorithms. Independent reconstruction uses
float64 or complex128. The script downloads public artifacts without an
authentication token and does not upload inputs. Weights and downloaded
upstream Python are not redistributed in this repository.

## Pinned models and reviewed source

The S4 weights come from
[krandiash/sashimi-release](https://huggingface.co/krandiash/sashimi-release/tree/83302f50c90d48d4b9ed4896683eacc4b47a824f),
revision `83302f50c90d48d4b9ed4896683eacc4b47a824f`, file
`checkpoints/sashimi_sc09_v4.pt` (about 19 MB). This is the SaShiMi release
for Speech Commands digit-audio modeling, linked by the official S4
repository. It is loaded with `torch.load(..., weights_only=True)`.
The release card does not specify a weights license; consult the authors'
release terms for reuse. This example observes the first S4 kernel,
`model.c_layers.0.layer.layer.kernel`, rather than running the entire audio
model. The direct output term `D` is omitted in both of its compared paths.

The numerical implementation is the official standalone
[S4 source](https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/models/s4/s4.py),
at commit `e757cef57d89e448c413de7325ed5601aceaac13`. The program checks its
SHA-256 before import. A narrowly specified adapter replaces only the
`rank_zero_only` distributed-logging decorator with an identity decorator,
avoiding a training-framework dependency. All numerical code is unchanged.
The run records both the original source hash and the adapter hash. The
naive CPU Cauchy backend is deliberate; no CUDA speed claim is made.

The complete DNA model is
[LongSafari/hyenadna-tiny-1k-seqlen-hf](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf/tree/e8c1effa8673814e257e627d2e1eda9ea5a373f6),
revision `e8c1effa8673814e257e627d2e1eda9ea5a373f6`, with a BSD-3-Clause
model card and a roughly 1.8 MB safetensors file. It has two width-128
blocks and 438,144 unique predictor parameters. The file also contains
non-parameter positional buffers. Each order-two operator has one long
convolution. The inspected model is the tiny 1k checkpoint, not the
million-position model in the same family.

HyenaDNA uses custom Python. We reviewed its three small upstream files,
pin them by revision and SHA-256, and import only those exact cached files:
`configuration_hyena.py`, `modeling_hyena.py`, and `tokenization_hyena.py`.
There is no unrestricted `trust_remote_code` loader. The source hashes in
the script are an explicit allowlist, not a replacement for reviewing a
future changed implementation. Safetensors stores the shared sinusoidal
frequency parameter once per block; four missing alias names are allowed
only after checking that all aliases refer to the same parameter object.
There are no other missing or unexpected weight keys.

## Fixed probes and numerical comparisons

Channels 0, 1, and 2 (printed as 1--3) were selected before viewing outputs.
S4 has 256 channels, state width 64, and 32 stored complex coordinates per
channel. Its shared continuous dynamics have channel-specific learned
step sizes and readouts. The 512-position diagnostic input is zero except
for amplitudes `1, -0.5, 0.75, 1.25` at zero-based positions
`0, 128, 256, 384`. This is a feature-level probe, not a speech sample.

An independent reconstruction expands conjugate pairs and builds
`A = diag(lambda) - P P*`, including the low-rank correction. It applies
the bilinear rule with linear solves, then recovers the recurrent readout
from the stored length-1000 corrected readout. Ignoring that correction
would compare different systems. All 512 native filter taps in three
channels agree with the independent complex128 construction within
`8.65e-8`. Native recurrent stepping agrees with the independent direct
convolution within `1.77e-7`; padded float64 FFT and direct convolution
agree within `1.12e-16`. These are observed errors; assertions allow
`1e-5`, `1e-5`, and `1e-12`, respectively.

All 64 eigenvalues of the full discrete transition are reconstructed for
each selected channel and lie inside the unit circle. The pole figure
plots these full eigenvalues, not the stored diagonal part. Its half-life
is `log(1/2) / log(abs(pole))`. Modes are sorted independently within each
channel, and their magnitudes do not measure task-level recall.

The DNA probe repeats `ACGTTGCAAGTCGATCCGTA` and takes the first 512
characters, without added special tokens. The exact string and token IDs
are in the run record. At zero-based position 255, C is replaced with A;
all other tokens and the sequence length remain unchanged. It is a
synthetic computational probe, not a biological sequence interpretation.

Hooks capture native block-1 short-filter branches and the result before
its output projection. For all 128 channels, independent direct summation
of `a * (conv(k, g * v) + D * g * v)` agrees with the native FFT path
within `5.88e-7` (assertion tolerance `3e-5`). The first 48 positions of
channel zero supply the gate figure and its exact frozen-gate matrix.
That matrix acts on the projected branch `v`; it is neither an attention
probability matrix nor a Jacobian through the gates and tokenizer.

The perturbation figure compares complete-model logits at all 512
positions, taking the largest absolute difference across all 16 output
coordinates, including padded vocabulary entries. The maximum earlier
prefix change is `6.68e-6`, consistent with FFT roundoff; the maximum
change at or after the altered input is `0.911`. Recomputing a separate
255-position prefix agrees with the full run's prefix within `8.59e-6`.
Both prefix checks have an absolute tolerance of `2e-5`. The expanded
prefix panel displays the small residual differences explicitly.

## Six figures

| File | What is observed |
| --- | --- |
| `ch13_pretrained_s4_filters.png` | Native taps in three fixed channels, each with its own vertical scale |
| `ch13_pretrained_s4_poles.png` | Full-transition poles and their envelope half-lives |
| `ch13_pretrained_s4_execution.png` | Native recurrent output, independent convolution, and carried states |
| `ch13_pretrained_hyena_filters.png` | Native implicit filters in three fixed channels |
| `ch13_pretrained_hyena_gates.png` | Gates, intermediate activations, and a frozen-gate coefficient matrix |
| `ch13_pretrained_hyena_causality.png` | Complete-model response to one changed future nucleotide |

The S4 state heatmap uses a logarithmic magnitude scale with a lower
display limit of `1e-7`, without normalizing rows. The gate matrix uses one
symmetric signed scale. All measured arrays are saved before plotting.
No figure measures model quality, biological relevance, or hardware speed.
