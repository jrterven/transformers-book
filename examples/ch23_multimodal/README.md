# Chapter 23. Vision–Language and Multimodal Models

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch23_multimodal.run_examples --output output/ch23.json
python -m unittest discover -s tests -p test_multimodal.py -v
```

## Score image-text pairs

`contrastive_logits` normalizes each nonzero feature row to unit
length and multiplies all image-text dot products by a positive
scale. Images index rows and texts index columns. With one paired
text for each image in a square batch, the positives lie on the
diagonal. `symmetric_contrastive_loss` returns cross-entropy across
rows, across columns, and their mean.

Normalizing an image row of scores gives weights over the supplied
text candidates. These weights depend on which candidates enter
the calculation. A language model generates from a separate
distribution over its vocabulary.

## Evaluate both retrieval directions

`bidirectional_recall_at_k` takes a binary relevance matrix with
the same shape as the scores. For each query it counts its relevant
items in the first `k` ranks, then divides by the number of relevant
items for that query. The reported mean weights each defined query
equally. A query with no relevant items returns `None` and is excluded
from the mean. A direction with no defined queries has mean `None`.
Score ties keep the original candidate order.

Some image-text protocols instead report the fraction of queries
with at least one relevant item in the first `k` ranks. State the
convention when comparing results with multiple positives.

## Turn visual features into language-width vectors

`image_feature_bridge` uses supplied query rows to read image keys
and values with Chapter 4's scaled dot-product attention. A
projection maps each resulting context row to the language-model
width. The example's query weights the two image values by
`(0.25, 0.75)`, then projects `(0.5, 3)` to `(0.5, 3, -2.5)`.
The small arrays expose the bridge calculation with fixed values.

## Trace the distinction

Equal feature widths permit a dot product; paired training aligns the
representations. Positive feature rescaling disappears under row
normalization. Positive logit rescaling preserves rankings while changing
softmax probabilities. Candidate duplication can change those probabilities
without changing the image or its scores for the original candidates.

The visual bridge changes two dimensions separately: query count sets the
number of output rows, while the final projection sets their feature width.
Projected rows are continuous inputs, not vocabulary IDs. Freezing the
language model's weights still permits derivatives with respect to its
inputs when training the bridge.

For paired visual tests, report both image accuracy and the fraction of
pairs with both answers correct. Merely changing answers can include
changing in the wrong direction. Keep ambiguity handling and unsupported
claims as separate judgments.

See the [11 exercises](EXERCISES.md) and [complete solutions](../../solutions/ch23_multimodal/README.md).
