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
