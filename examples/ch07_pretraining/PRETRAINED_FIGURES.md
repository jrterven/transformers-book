# Inspecting pretrained GPT-2 and BERT

The expanded chapter follows actual tokenization, input vectors, vocabulary
predictions, and controlled context changes in two published predictors. These
six figures are measured outputs, not illustrations of invented probabilities.
They complement the small objective examples in this directory, which continue
to work without pretrained weights or the optional Transformers dependency.

Run from the repository root:

```sh
python -m pip install -r requirements-ch07-figures.txt
python -m examples.ch07_pretraining.pretrained_figures \
  --cache-dir /path/to/huggingface/hub \
  --output-dir output/ch07-pretrained
```

The first run downloads public, pinned safetensors weights, with authentication
disabled and `trust_remote_code=False`. Allow roughly 1 GB of download/cache
space for the two checkpoints, plus runtime memory. Once the cache contains
both models and tokenizers, add `--offline` to prohibit downloads. The cache
argument is the Hugging Face hub cache itself, rather than its parent folder.
No training data or manuscript files are uploaded.

## Checkpoints and execution

| Predictor | Pinned revision | Vocabulary | Loaded parameters |
| --- | --- | ---: | ---: |
| [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 50,257 | 124,439,808 |
| [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) | `86b5e0934494bd15c9632b12f734a8a67f723594` | 30,522 | 109,514,298 |

Both loaded predictors have 12 blocks, 12 attention heads per block, and width
768. GPT-2 has 1,024 learned absolute positions; this BERT has 512. Equal width
and depth do not make their training, tokenizers, or block internals identical.
The model cards label GPT-2 MIT and BERT Apache 2.0. No weights are redistributed
in this repository. Their observed weight-file SHA-256 hashes are:

```text
GPT-2  248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707
BERT   68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3
```

Execution uses CPU float32, one PyTorch thread, deterministic algorithms,
evaluation mode, frozen parameters, and eager attention so native probabilities
are returned. No sampling or training occurs. All required predictor weights
must load without missing, mismatched, or newly initialized keys. BERT's original
pooler and next-sentence classifier are intentionally unused by
`BertForMaskedLM`; their four unused checkpoint keys are recorded. The MLM
head is fully pretrained. GPT-2 uses `GPT2LMHeadModel` and has no unused keys.

The prompts, feature coordinates 0--7, block 1/head 1, and displayed hidden
stages 0, 1, 4, 8, and 12 are fixed in the source before inspection. All 13
hidden stages, all 12 heads in every block, full vocabulary logits, and the
full feature arrays are saved. Stage 0 is the model's embedding output; GPT-2's
stage 12 includes its final layer normalization. Distances across stages need
not increase and do not measure semantic importance.

## What each figure shows

| File | Measured computation |
| --- | --- |
| `ch07_gpt_prediction.png` | Five actual prefix IDs; lookup versus final contextual features; eight next-token probabilities from the complete 50,257-entry softmax |
| `ch07_gpt_generation.png` | Three manual greedy feedback steps, recomputing the complete prefix each time, with no suppressed classes |
| `ch07_gpt_causal_context.png` | Native first-head attention and all-feature hidden differences after changing only `France` to `Italy` |
| `ch07_bert_inputs.png` | All 13 WordPieces, IDs, positions, and segment IDs for two segments; three input contributions, their sum, and native input LayerNorm |
| `ch07_bert_mask_predictions.png` | Native `[MASK]` predictions in `The [MASK] can bark.` versus `The [MASK] can fly.` |
| `ch07_bert_bidirectional_context.png` | Native first-head attention and all-feature hidden differences after changing only `bark` to `fly` |

The pair is `An unaffordable car.` and `It stopped.`. The second segment's
`it` occupies array position 9 and has segment ID 1. WordPiece splits
`unaffordable` into `una`, `##ff`, `##ord`, and `##able`. The input table's colors
identify segment membership; they impose no attention prohibition. Input
LayerNorm statistics use all 768 coordinates, not just the eight shown.

Every probability bar retains its original full-vocabulary value. The sum of
unshown probabilities is labeled separately. A middle dot displays a literal
space in GPT-2's decoded pieces. This notation does not add an input character.
Both context-map panels label their own actual input tokens. The lower distance
maps use the original sequence as the position labels and an independent
distance scale. All native attention maps share the probability range 0--1.

## Observed results and checks

With `The capital of France is`, GPT-2 assigns approximately 0.084591 to the
space-prefixed `the` and 0.032244 to the space-prefixed `Paris`. Greedy feedback
produces `The capital of France is the capital of` after three new tokens,
without an end-of-text token. This trace demonstrates repetition in a raw
continuation model. It is not presented as a successful factual answer.
The three-step cap is the stopping reason; a period is not GPT-2's end token.

Teacher-forced scoring of the reference `The capital of France is Paris.`
has six shifted labels and mean NLL about 3.798257. Scoring only its final
two continuation labels, ` Paris` and `.`, gives mean NLL about 2.388457.
The separate conditional probabilities are about 0.032244 and 0.261196;
their product is about 0.008422. The prompt supplies context in both cases,
but the evaluated label counts differ. The full shifted loss is checked
against the model's built-in causal loss.

Changing `France` to `Italy` changes exactly one input ID at position 3.
Every earlier hidden vector at all 13 returned stages is unchanged in the
checked run, as are earlier logits. All heads and layers have exactly zero
attention above the diagonal. The later final position can change; ` Rome`
becomes its highest-probability candidate, about 0.157346.

In BERT, `dog` is the top mask candidate for `can bark`, about 0.066576, and
`bird` is the top candidate for `can fly`, about 0.104184. The mask's input
embedding output is identical in both sequences. Its final 768-feature vectors
differ by Euclidean distance about 4.130264. This tests an actual right-context
effect; the architecture permits such effects without requiring every possible
edit to change every output. A single observed head is not a unique causal
explanation of a vocabulary choice.

Independent NumPy float64 calculations check the model outputs rather than
only plotting them:

- GPT-2's tied vocabulary projection is reconstructed from its complete final
  vector and token table, using chunked products over all vocabulary entries.
  Maximum logit discrepancy is below 0.0002; a stable independent softmax is
  also checked against native probabilities.
- BERT's word, position, and segment sums and population-variance LayerNorm
  reconstruct the native embedding output within `1e-6` in the recorded run.
- BERT's dense MLM transform, exact GELU, separate LayerNorm, tied token
  projection, and vocabulary bias reconstruct every mask logit within `2e-5`.
- Attention rows sum to one; GPT-2's future entries are exactly zero; the
  controlled edits change only their intended token IDs. Distance plots use
  all 768 features, not the small feature excerpt shown elsewhere.

The source contains explicit tolerances, including small relative allowances
for native float32 projections. Raw loss values, probabilities, and feature
distances from the two models do not constitute a model-quality comparison.
There is no held-out benchmark in these demonstrations.

## Saved record

[`PRETRAINED_FIGURES_RUN.json`](PRETRAINED_FIGURES_RUN.json) records the November 9,
2026 run: exact checkpoint revisions, weight hashes, source hash, software,
input IDs, probabilities, numerical discrepancies, unused keys, and artifact
hashes. The command writes a new record beside six PNGs and
`observations.npz`, which contains the complete numeric arrays. Generated
arrays and images remain local; only the compact record and reproducible
source are versioned here. PNG and archive hashes can vary with plotting or
compression software; inspect the numerical record and checks when reproducing
on another platform.

The original architecture and objectives are described in the
[GPT report](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf),
[GPT-2 report and code](https://github.com/openai/gpt-2), and
[BERT paper](https://aclanthology.org/N19-1423/).
The observed implementation is Transformers 4.57.1:
[GPT-2 source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gpt2/modeling_gpt2.py)
and [BERT source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/bert/modeling_bert.py).
