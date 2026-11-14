# Measured language understanding examples

Redraw the six figures from the [recorded model outputs](PRETRAINED_RUN.json).

```sh
python -m pip install -r requirements-ch16-figures.txt
python -m examples.ch19_understanding.pretrained_figures
```

Run the checkpoints again on a CPU with the following commands.

```sh
python -m examples.ch19_understanding.pretrained_observations \
  --cache-dir output/model-cache --output output/ch19-run.json
python -m examples.ch19_understanding.pretrained_figures \
  --record output/ch19-run.json --output-dir output/ch19-real
```

Use `--offline` after caching the checkpoints. If the base BERT is already in
another cache, specify its directory with `--bert-cache-dir`. The program loads
safetensors, uses local Transformers classes, and stores the exact checkpoint
revisions and SHA-256 digests. The reference run uses four CPU threads and
float32 with eager attention.

## BERT representations and pooling

The [BERT checkpoint](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594)
encodes four sentences that share the same prefix and differ after *bank*.
The record contains the cosines between its four representations at every
layer. It verifies that all four embedding rows are equal before the blocks.

Two linear heads train on the 96 fictional requests from Chapter 16. The
tokenizer sees each request without an instruction. The heads use the final
`[CLS]` row or the mean of the final content rows. Both use training-only
standardization, a zero start, and 100 fixed AdamW updates. Both are scored
on the same 32 requests with disjoint accounts and wording families. The
padding probe on the first held-out update compares changes to the attention
mask and the pooling mask separately.

## Sentence-pair inference

The [inference checkpoint](https://huggingface.co/cross-encoder/nli-distilroberta-base/tree/b14d131f9d32668a5e6a982729b57ff6ed5dfcbd)
returns contradiction, entailment, and neutral scores in that order. Each of
the six examples keeps both input sentences, the packed token IDs, raw scores,
softmax probabilities, and a reference label. The swapped-caller example
has a neutral reference and a high entailment score.

## Named entities and capitalization

The [entity checkpoint](https://huggingface.co/dslim/distilbert-NER/tree/dfa2838a127384aabb82ed7719e16dab84c42a2a)
scores 14 authored sentences and their 14 lowercase counterparts. Word
boundaries and gold spans stay fixed. Each input is tokenized separately
with the supplied word boundaries, and each word's first piece supplies its
prediction. The primary span metrics interpret the selected tags under IOB2
with optional repair of misplaced inside tags. Word accuracy compares the
original first-piece tags. The constrained decoder also finds the highest
scoring legal IOB2 path and saves its tags and spans separately.

`PRETRAINED_RUN.json` includes the tokenizer word IDs, native token logits,
gold tags, first-piece targets, and decoded spans. The original and lowercase
sets have 93 word and punctuation positions and 26 gold entities each. The
figures measure this fixed example set and preserve the per-case outcomes.
