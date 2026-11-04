# Chapter 1: Tokens, Embeddings, and Neural Language Models

This example follows **the red car** from token IDs to vectors and a next-token
probability distribution. It uses NumPy and small matrices chosen by hand.
The weights were not trained on text. Their predictions illustrate arithmetic,
not language understanding or the quality of a language model.

From the repository root:

```sh
python -m pip install -r requirements.txt
python -m examples.ch01_language.run_examples
python -m unittest discover -s tests -v
python scripts/export_snippets.py --chapter ch01 --check
```

The first two commands run on a CPU with NumPy alone. The complete repository
suite also contains optional PyTorch tests for Chapters 5 and 6; those are
reported as skipped when PyTorch is unavailable. No corpus or model download is
needed.

## Follow the calculation

The vocabulary has 16 entries. PAD, BOS, and EOS use IDs 0, 1, and 2. The example
assigns **the** ID 4, **red** ID 12, and **car** ID 7. An ID chooses a row; its
numerical magnitude is not a measure of meaning or similarity.

The embedding table `E` has shape `(16, 4)`. Its three displayed rows are:

```text
the:  ( 0.2, -0.4,  0.1,  0.5)
red:  (-0.3,  0.8,  0.2, -0.1)
car:  ( 0.6,  0.1, -0.5,  0.3)
```

The other rows are zero in this controlled example. A trained embedding table
would generally learn different values for all vocabulary entries. Looking up
the same ID retrieves the same vector each time. The later context calculation
combines those vectors in their current order.

The small network reads only the last two tokens of a prefix. It concatenates
their vectors, applies an affine map and ReLU, then projects the resulting
hidden vector into one score per vocabulary entry:

```text
context IDs:          (2,)
looked-up vectors:    (2, 4)
concatenated vector:  (8,)
hidden vector:       (4,)
logits/probabilities: (16,)
```

`W_context` stacks an identity matrix above twice that identity. Its bias is
`(0, -0.2, 0, 0)`. Thus **the red** gives hidden vector `(0, 1, 0.5, 0.3)` after
ReLU. The output matrix reads its second coordinate: the **car** score is 2,
the **red** score is 1, and the other fourteen scores are 0.

Softmax normalizes all sixteen entries:

| Entry | Probability after **the red** |
| --- | ---: |
| car | 0.306506513543 |
| red | 0.112757444918 |
| Each other vocabulary entry | 0.041481145824 |

The denominator is `exp(2) + exp(1) + 14`, not the denominator of a three-class
softmax. PAD and BOS remain vocabulary entries in this raw forward-pass
illustration. No generation rule suppresses their scores here. The probabilities
sum to 1, and **car** is the most probable individual next token. The loss for
that correct target is `-log(P(car)) = 1.1825162724674187`.

Reversing the two input tokens to **red the** gives hidden vector
`(0.1, 0, 0.4, 0.9)`. All output logits are then zero, producing probability
`1/16` for each entry. The difference shows that this calculation is sensitive
to token order. These hand-chosen weights do not prove that a model understands
the two phrases.

## Use the functions

`toy_parameters()` returns the example matrices. The following functions keep
the shapes visible:

- `embedding_lookup(E, ids)` appends the vector-width axis to the ID shape. IDs
  of shape `(3,)` give vectors `(3, 4)`; a batch `(2, 3)` gives `(2, 3, 4)`.
- `context_logits(context_ids, parameters)` expects exactly two IDs on the last
  axis. It returns hidden vectors and logits, with leading batch axes preserved.
- `stable_softmax(logits)` normalizes only the last axis. It subtracts each
  row's largest score before taking exponentials.
- `token_nll(logits, targets)` returns one negative log-probability per target.
  It uses stable log-probabilities, avoiding `log(0)` from rounded or underflowed
  probabilities.
- `next_token_pairs(sentence_ids)` makes two-token training contexts and their
  following labels for a nonempty sentence of ordinary word IDs.

For **the red car**, the next-token pairs are:

```text
BOS BOS -> the
BOS the -> red
the red -> car
red car -> EOS
```

The repeated BOS tokens fill the missing left context for the fixed-width
network. They do not mean that a sentence normally contains two beginning
markers. An unrestricted language model can use the whole earlier prefix;
this small model deliberately retains only its last two inputs. EOS is a
prediction target; BOS and PAD are not targets in these pairs.

The five book listings are generated from these source files. After editing a
marked block, run `python scripts/export_snippets.py --chapter ch01`, then use
`--check` to verify the excerpts.
