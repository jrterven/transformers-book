# Actual pretrained-model adaptation

From the companion repository root, preferably in a dedicated Python environment:

```sh
python -m pip install -r requirements-ch16-figures.txt
python -m examples.ch16_adaptation.pretrained_figures --train \
  --cache-dir output/model-cache --output-dir output/ch16-pretrained
```

The first run obtains the pinned public base. Add `--offline` once cached.
Omit `--train` to redraw saved observations. The outputs include six PNGs,
`observations.npz`, three `.safetensors` component checkpoints, `run.json`,
and the compact `public-run.json`. The [recorded run](PRETRAINED_FIGURES_RUN.json)
contains all messages, generations, settings, hashes, and checks. Native model
weights are not redistributed in this code repository.

## Published source and numerical execution

The base is [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2),
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, with an Apache-2.0
model card. The native Transformers 4.57.1 Llama implementation loads every
required weight without missing, unused, or mismatched keys. Execution uses
PyTorch 2.15.0, CPU SDPA, float32, four threads, seed 16, deterministic
algorithms, and evaluation-mode forward behavior. Autograd remains enabled
where required. No remote model code, quantization, or external training data
is used. Generated continuations are greedy, limited to eight new tokens.

The base has 134,515,008 distinct parameter scalars, hidden width 576, 30
blocks, nine query heads, and three KV heads. All pretrained tensors remain
unchanged during both training stages, checked by a digest over their values.
Their operations remain in the backward path for adapters. The script changes
the *effective model* through adapters; frozen base tensors do not imply
unchanged predictions or preserved unrelated abilities.

## Data and fixed protocol

All requests, account identifiers, and addresses in `real_data.py` are
fictional. There are 96 training messages (six templates per class, eight
accounts) and 32 held-out messages (four different templates per class, four
different accounts). Both template families and accounts are disjoint.
The two classes have equal example counts. No held-out outcome selects the
checkpoint, optimizer step, or hyperparameters. One fixed seed illustrates
mechanisms; this is not a general benchmark or a multi-seed estimate.

The common prompt ends in `Intent:`. The full response is ` LOOKUP` or
` UPDATE`, plus native EOS. The tokenizer gives three and two content pieces,
respectively, so supervised target counts including EOS are four and three.
The program asserts that separate prompt/response encodings equal their
concatenation. Existing token 1 is padding only; EOS is token 0 and is scored.
Loss labels are shifted with the response mask. Batches are right-padded for
training; generation is left-padded with an explicit attention mask.

## Frozen-feature head

The last prompt token's final pretrained hidden vector supplies 576 features.
Coordinate means and population standard deviations come from training
features only; standard deviations are floored at `1e-4`. The same fixed affine
transform applies to held-out features. It has 1152 stored, nontrainable
scalars; the two-class head has 1154 trainable scalars. Zero-initialized head
weights/biases are trained for 100 full-batch AdamW steps, learning rate `.01`,
weight decay `.001`. Training accuracy is 96/96 and held-out accuracy 27/32.
The checkpoint includes head weights, bias, and normalization statistics.

## Actual SFT/LoRA training

Rank-four, alpha-eight adapters are attached to query and value projections
in all 30 blocks, with exactly 230,400 trainable scalars. Native query matrices
are `[576,576]`; value matrices are `[192,576]`. The base and every other
parameter remain frozen. Random A and zero B preserve starting logits exactly.
The first A gradients are zero and B gradients nonzero; both learn afterward.

The run takes 64 AdamW steps, batches of eight, learning rate `.002`, zero
weight decay, and gradient-norm clipping at one. The response loss is the mean
over selected tokens in each batch, including EOS. This differs from giving
all responses equal weight. The held-out token NLL changes from about 2.495 to
.531. Comparing the summed probabilities of two complete candidate labels
improves from 20/32 to 28/32; this is a constrained two-string comparison,
not full-vocabulary generation or a classification head.

Separately, exact greedy label-only text (after trimming outer whitespace)
changes from 0/32 to 28/32. The base often begins with a label but continues a
new example. All 32 continuations at each stage are recorded, including
failures. Learning to emit EOS is part of response-only SFT.

## Actual DPO training

The frozen reference is the SFT adapter checkpoint. Forty-eight training pairs
use training templates and four training accounts. The preferred completion is
the bare correct label; the rejected completion is an explanatory sentence
with the same class. The preference follows the label-only instruction. These
are authored format preferences, not human general-quality judgments.

Reference chosen/rejected scores are cached once with fixed formatting and
EOS handling. Their digest stays unchanged. The policy takes 16 AdamW steps
with batch size four, learning rate `.0002`, zero weight decay, clipping at
one, and beta `.1`. The held-out mean reference-adjusted gap is about 57.27
natural-log units and pair loss about `.00353`. Every chosen completion's
absolute log-probability decreases, while rejected scores decrease much more.
The ratio improves, even as supervised held-out token NLL rises to about .656.
Greedy exact-label outputs remain 28/32. This is a measured example of different
objectives moving differently; it makes no general claim about DPO's quality.

## Saved and measured checks

- Actual autograd: ignored logit rows have exactly zero direct loss gradient,
  while earlier prompt embeddings have nonzero derivatives for later response
  predictions. Differentiating inputs does not update the embedding table.
- An actual block-30 query update has numerical rank four when learned factors
  are multiplied in float64. Its small singular-value tail is roundoff; the
  restriction applies to the update, not the full pretrained matrix.
- SFT adapter reload reproduces the recorded first-token distribution exactly;
  DPO reload reproduces all checked logits exactly in this CPU replay.
- Merging all 60 trained adapters reproduces two held-out teacher-forced
  sequences within maximum logit discrepancy `0.0001373291015625` over all
  positions/vocabulary entries, with identical argmax decisions. The acceptance
  tolerance is `2e-3`, separate from that observed value.
- Each adapter checkpoint is 933,544 bytes, including header and names. Actual
  factor payload is 921,600 bytes; neither figure is peak training memory.
- Two complete final training runs reproduce the same metrics, generations,
  component-checkpoint hashes, and six final image files. Platform, exact
  versions, base/runtime hashes, and all module paths are recorded.

The classifier and generator have different interfaces and objectives. Their
27-versus-28 held-out counts are not a controlled comparison of update
parameterizations. No unrelated-task preservation, general alignment quality,
optimized-GPU throughput, or peak-memory advantage is measured.
