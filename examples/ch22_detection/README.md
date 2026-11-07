# Chapter 22. Object Detection and Segmentation

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch22_detection.run_examples --output output/ch22.json
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

See the [11 exercises](EXERCISES.md) and [complete worked solutions](../../solutions/ch22_detection/README.md),
including all explanations and three plots in both PDF editions. Prompting
protocols distinguish supplied annotated locations from detector boxes,
with misses and extra detections reported explicitly.
