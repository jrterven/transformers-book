# Chapter 19. Translation and Summarization

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch19_generation.run_examples --output output/ch19.json
python -m unittest discover -s tests -p test_generation.py -v
```

## Shift and pad a batch of text pairs

`collate_pairs` takes unpadded source IDs and target content IDs. Add
source boundary tokens beforehand if the tokenizer requires them.
The function prepends BOS to decoder input and appends EOS to
the target labels. A target with no content still has one label,
EOS, predicted from BOS. Source and target arrays are right padded
to their respective widths.

The optional CPU step reuses the Chapter 5 Transformer and the
Chapter 6 training step and padding-aware loss. It measures the
same-batch loss before and after a single update of a small
randomly initialized model.

```sh
python -m pip install -r requirements-ch05.txt
python -m examples.ch19_generation.run_examples --train-step --output output/ch19-training.json
```

## Keep several possible continuations

`beam_search` takes a callback that returns a normalized mapping
from token IDs to next-token probabilities for a generated prefix.
The callback can close over a fixed source. It receives the empty
prefix on the first call. Search adds log probabilities and keeps
the highest scoring candidates after each step. Equal scores are
ordered by token IDs. Completed candidates retain their scores
and stay eligible for selection. A result that reaches the cap
without EOS is marked unfinished.

Width one gives the greedy result. In the fixed example, it keeps
A and then EOS, for probability 0.33. Width two keeps B available
long enough to find B followed by EOS, for probability 0.38.

## Inspect overlap with a reference

`clipped_ngram_precision` limits each candidate n-gram count by the
reference count. Its denominator counts all candidate n-grams at the
specified order. A candidate shorter than that order returns `None`.
`rouge_l` reports an LCS length and per-example precision, recall,
and F1. An empty candidate or reference has zero overlap scores.
Provide token lists using the same tokenization for both sides.
Inspect source coverage and unsupported additions alongside these
reference overlap measures.
