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

## Reproduce the pretrained demonstrations

The gallery uses five photographs and eight English descriptions. Three
image rows have two valid descriptions. The spaceflight description has
two relevant photographs, and the sailboat description has none. Its
ranking is recorded and its recall is undefined. The [frozen run](PRETRAINED_RUN.json)
contains the relevance matrix, cosine matrices, complete rankings, macro
recall, and hit counts in both directions.

The two retrieval models also rank left/right captions for the original
and mirrored spoon and flag photos. Only the direction word changes in a
caption pair. The trace keeps both cosines per image and records the top
caption direction before comparing with the answer key. This is a forced
ranking over two supplied descriptions. The visual assistant instead
generates from its vocabulary on the same two photo pairs.

- [SigLIP 2 Base](https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2)
  uses 768-wide unit rows. The run checks that its original pair logits
  equal the cosine times a learned scale plus a learned bias.
- [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/tree/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda)
  encodes each image or text separately with the same system instruction.
  It pools the last non-padding row and normalizes all 2,048 coordinates.
  This example uses native Transformers classes and the checkpoint's
  image processor. Its inputs and rankings are independent of SigLIP 2.
- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)
  generates answers for four pairs of original and horizontally mirrored
  photographs. It also answers the same questions with the images removed.
  The runner keeps the exact questions and decoded strings, counts correct
  views and complete pairs, and records the first-step logits for the two
  single-token forms ` left` and ` right`. The two-token share is conditional
  on those alternatives. The record includes the right token's separate
  probability over the full vocabulary.

Use Python 3.12. The optional CPU run downloads about 7.5 GB of model
weights the first time. It runs one checkpoint at a time and verifies the
weight digests. Pass `--experiment siglip`, `--experiment embedding`, or
`--experiment visual` to run one part. Pass `--offline` to require cached
weights and photographs. The figure command defaults to the frozen run.
It uses only NumPy, Pillow, Matplotlib, and the photographs.

```sh
python -m pip install -r requirements-ch23-figures.txt
python -m examples.ch23_multimodal.pretrained_observations
python -m examples.ch23_multimodal.pretrained_figures
```

To render a newly measured run, pass `--record output/ch23-pretrained.json`
to the figure command. Numerical results may vary slightly across CPU
implementations. The questions, labels, and evaluation set are fixed by
[`demo_data.py`](demo_data.py). The collection illustrates behavior on a
small set of examples, with four paired questions from two source photos.

Photographs come from the [scikit-image example data](https://scikit-image.org/docs/stable/api/skimage.data.html).
Coffee is by Rachel Michetti, Chelsea by Stefan van der Walt, and Camera
by Lav Varshney (CC0). The Eileen Collins portrait is by NASA and the
Falcon 9 photograph by SpaceX (public domain). The figures resize, pad,
or mirror the credited photographs where indicated.

See the [11 exercises](EXERCISES.md) and [complete solutions](../../solutions/ch23_multimodal/README.md).
