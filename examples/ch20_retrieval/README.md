# Chapter 20. Search, Retrieval, and Grounded Assistants

Run the local examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch20_retrieval.run_examples --output output/ch20.json
python -m unittest discover -s tests -p test_retrieval.py -v
```

## Compare lexical and dense scores

`bm25_scores` tokenizes query and passages into lowercase words,
including numbers, and scores each distinct query term once. A
term's contribution uses document frequency, saturation with term
frequency, and the passage length relative to the corpus mean.
`dense_scores` calculates dot products against supplied vectors.
The example vectors are chosen by hand.

Both score arrays follow the input passage order. `rank_passages`
takes the IDs in that same order and sorts by score, with passage
IDs breaking ties. The lexical top two are P2 and P1. The example
supplies pairwise relevance scores for just those candidates and
`rerank_candidates` places P1 first.

## Pack passages and trace evidence

`pack_context` visits the ranked IDs in order and keeps a passage if
its whole text fits in the remaining budget. It can skip a longer
passage and consider later ones. The teaching budget counts words
in passage text. IDs are preserved as separate fields. A deployed
system can measure its complete formatted prompt with the model's
tokenizer.

`literal_evidence_hits` checks for consecutive word tokens within a
passage. The answer fixture cites the mobility desk in P1. The
permit-price fixture has an empty set of annotated evidence IDs and
leaves the answer blank. Literal occurrence and human judgments of
answer support are separate checks. `evidence_recall` reports the
fraction of annotated relevant IDs in the requested top ranks and
returns `None` when the annotation set is empty.

## Work through the exercises

The [12 exercises](EXERCISES.md) and [complete solutions](../../solutions/ch20_retrieval/README.md)
include a written PDF, executed notebook, and notebook PDF. Every edition
contains all solutions and two plots. The RAG-mixture calculation uses one
fixed prefix and original retrieved set, exposing why averaging before or
after multiplying changes its probability.

The supplied dense vectors have unit norm, but the helper computes raw dot
products. Scores across lexical and dense systems are not on a common scale.
Packing is a greedy whole-passage rule, not an evidence-coverage optimum.
The worked budgets use passage-text word counts. For a decoder-only generator,
reserve output space in the shared context window and count all formatted
prompt tokens; encoder-decoder input and output limits are separate.

Audit source authority, effective intervals and exact versions alongside the
claim. A later upload need not govern a current question. Track evidence at
retrieval and packing separately, and report false abstentions on answerable
questions separately from unsupported answers on unanswerable questions.

## Run the released models

The measured example searches a fictional archive of 16 campus notices.
It records lexical and Qwen3 embedding scores, rank fusion, Qwen3
reranking, and the exact Qwen3.5 answers. The renewal probe follows the
same question through a lexical miss, a tight prompt budget, and an
input containing just the annotated evidence.
[Reproduce the runs and figures](PRETRAINED_FIGURES.md).
