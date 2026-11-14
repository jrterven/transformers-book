# Chapter 23. Object Detection and Segmentation

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch23_detection.run_examples --output output/ch23.json
python -m unittest discover -s tests -p test_detection.py -v
```

## Compare boxes and masks

`box_iou` takes positive-area boxes in `[x0, y0, x1, y1]` corner
coordinates. Width and height are coordinate differences. Boxes
that touch only at an edge have zero overlap. `mask_iou` counts
foreground pixels on a common Boolean grid. Two masks with an
empty foreground union return `None`.

## Assign one query per target

`match_queries` takes a cost matrix with query rows and target
columns. It enumerates assignments with distinct query indices,
computes their total costs, and takes the lowest. It returns
zero-based `(query, target)` pairs in target order, unmatched
queries, and total cost. Ties use lexicographic query order. The
exhaustive search is for small matrices and stops at a cap of
100,000 assignments. A matrix with zero target columns leaves
every query unmatched.

## Score detections in confidence order

`precision_recall` considers predictions from one class and one
image. It visits predictions in descending confidence order. A
prediction can match the highest-IoU unused target if its IoU is
at least the threshold. Each target can be matched once. Score
ties preserve input order, and IoU ties choose the first target.

`all_points_ap` integrates the backward precision envelope over
recall increments at the one supplied IoU threshold. The worked
example has one duplicate prediction and AP of `5/6`. With
targets and no predictions, AP is zero. With no targets, recall
and AP are `None`. The book discusses the additional protocol
choices involved in a COCO evaluation.

## Interpret the checks

Training minimizes a total assignment cost; ranked evaluation claims
annotations in descending prediction score. They use different matching
rules for different purposes. A one-to-one constraint does not guarantee
a unique optimum: the Exercise 3 matrix has two minima of cost four.

Continuous box edges have width `x1 - x0`, with no extra endpoint unit.
An empty mask union is undefined (`None`), while exactly one empty mask
has IoU zero. Undo padding before scale when returning predictions to the
original image. A box can remain in bounds despite using the wrong inverse.

All-points AP uses a backward precision maximum and recall increments.
Unattained recall contributes zero area. Confidence cutoffs filter
predictions; IoU thresholds determine acceptable overlap. Dataset AP pools
ranked predictions across images per class rather than averaging image AP.
False positives in images without that class must remain in the dataset
ranking. Use an official evaluator for COCO results.

## Reproduce the pretrained demonstrations

One COCO validation photograph (image `39769`, 640 × 480) follows two
cats and two remote controls through three released checkpoints:

- [RF-DETR Nano](https://huggingface.co/Roboflow/rf-detr-nano/tree/9f201577d9415f0378084748d27e68586e6b3600)
  outputs normalized boxes and sigmoid class scores. The trace keeps the
  original query and class for each ranked pair. The figures show the two
  classes above `0.50`, while the ranking example keeps remote scores above
  `0.04`. AP at `0.90` and `0.93` uses the same eleven predictions.
- [RF-DETR Seg Nano](https://huggingface.co/Roboflow/rf-detr-seg-nano/tree/472badbc863609789ce1f69311244c670bdfd2b9)
  adds per-query mask logits. Resize the logits to the original image
  with bilinear interpolation, then threshold at zero. The record keeps
  each binary map and its overlap counts against same-class annotations.
- [SAM 2.1 Hiera Tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny/tree/de431c4043854a71d8101e17995dfe596bf101a5)
  reuses the image embeddings across prompts. One foreground click at
  `(270, 209)` tests whether the highest quality score selects the mask
  for the whole left cat. Three box conditions compare annotated boxes,
  detected boxes, and a deliberate 96-pixel shift of only the left
  detector box. Each condition selects by the model's quality score
  before comparison with COCO.

The results are controlled demonstrations on one photograph. See the
[frozen run](PRETRAINED_RUN.json) for package versions, checkpoint revisions
and SHA-256 digests, tensor shapes, scores, and binary maps in compressed
COCO run-length encoding. Same-class overlaps are in left-to-right reference
order. Indices into the SAM candidate lists start at zero.

Use Python 3.12 and the optional dependencies to run inference on CPU.
The first run downloads the photograph and roughly 400 MB of model weights.
The renderer uses the frozen run by default and needs only the photograph
and the packages used for figures. Pass `--record output/ch23-pretrained.json`
to render a new run, or add `--offline` to require cached inputs.

```sh
python -m pip install -r requirements-ch23-figures.txt
python -m examples.ch23_detection.pretrained_observations
python -m examples.ch23_detection.pretrained_figures
```

The photograph is [*Cats and remote controllers* by DocChewbacca](https://www.flickr.com/photos/st3f4n/210383891/),
licensed [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/).
The four photograph-based figures add overlays or crops and are available
under the same license. [COCO's annotations](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)
are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The
[fixture](COCO_39769.json) contains the four relevant polygons and boxes,
and identifies the source and hash of the full annotation file.

See the [11 exercises](EXERCISES.md) and [complete worked solutions](../../solutions/ch23_detection/README.md),
including all explanations and three plots in both PDF editions. Prompting
protocols distinguish supplied annotated locations from detector boxes,
with misses and extra detections reported explicitly.
