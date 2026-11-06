"""Small NumPy examples for set matching, IoU, and ranked detections."""

from itertools import permutations
import math

import numpy as np


def _real_array(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return values


def _boxes(values):
    boxes = _real_array(values, "boxes").astype(np.float64, copy=False)
    if boxes.shape == (0,):
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes need shape [count, 4] in x0, y0, x1, y1 order")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("boxes must have finite corners and positive width and height")
    return boxes


# snippet:iou
def box_iou(a, b):
    """Compare positive-area boxes with continuous corner coordinates."""
    a, b = _boxes([a, b])
    with np.errstate(over="raise", under="raise", invalid="raise"):
        lo = np.maximum(a[:2], b[:2])
        hi = np.minimum(a[2:], b[2:])
        intersection = np.maximum(0, hi - lo).prod()
        area_a = (a[2:] - a[:2]).prod()
        area_b = (b[2:] - b[:2]).prod()
        union = area_a + (area_b - intersection)
    return float(intersection / union)


def mask_iou(a, b):
    """Compare Boolean masks on the same pixel grid."""
    a, b = map(np.asarray, (a, b))
    if a.ndim != 2 or a.shape != b.shape or a.dtype != np.bool_ or b.dtype != np.bool_:
        raise ValueError("masks must be Boolean arrays on the same 2D grid")
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return intersection / union if union else None
# end-snippet:iou


# snippet:match_queries
def match_queries(cost):
    """Enumerate assignments for a small cost matrix [queries, targets]."""
    cost = _real_array(cost, "cost")
    if cost.ndim != 2:
        raise ValueError("cost needs shape [queries, targets]")
    queries, targets = cost.shape
    if targets > queries:
        raise ValueError("one-to-one assignment needs at least as many queries as targets")
    if math.perm(queries, targets) > 100_000:
        raise ValueError("the teaching search allows at most 100,000 assignments")

    def total(chosen):
        return sum(cost[q, t].item() for t, q in enumerate(chosen))

    # Each candidate lists one distinct query for each target in order.
    chosen = min(permutations(range(queries), targets), key=total)
    pairs = tuple((q, t) for t, q in enumerate(chosen))
    unmatched = tuple(q for q in range(queries) if q not in chosen)
    return pairs, unmatched, total(chosen)
# end-snippet:match_queries


def _ranked_inputs(ground_truth, predictions, scores, threshold):
    ground_truth = _boxes(ground_truth)
    predictions = _boxes(predictions)
    scores = _real_array(scores, "scores")
    if scores.shape != (len(predictions),):
        raise ValueError("each prediction needs one score")
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be positive and at most one")
    return ground_truth, predictions, scores


# snippet:precision_recall
def precision_recall(ground_truth, predictions, scores, threshold=0.5):
    """Rank detections and match each target at most once within one image."""
    ground_truth, predictions, scores = _ranked_inputs(
        ground_truth, predictions, scores, threshold
    )
    order = np.array(sorted(range(len(scores)), key=scores.__getitem__, reverse=True),
                     dtype=np.intp)
    unused = np.ones(len(ground_truth), dtype=bool)
    true_positive = np.zeros(len(order), dtype=bool)
    for rank, prediction in enumerate(order):
        available = np.flatnonzero(unused)
        if available.size == 0:
            break
        overlaps = [box_iou(predictions[prediction], ground_truth[j]) for j in available]
        best = int(np.argmax(overlaps))
        if overlaps[best] >= threshold:
            true_positive[rank] = True
            unused[available[best]] = False

    cumulative = np.cumsum(true_positive)
    precision = cumulative / np.arange(1, len(order) + 1)
    recall = cumulative / len(ground_truth) if len(ground_truth) else None
    return order, true_positive, precision, recall
# end-snippet:precision_recall


# snippet:all_points_ap
def all_points_ap(precision, recall):
    """Integrate the precision envelope over increases in recall."""
    if recall is None:
        return None
    precision = _real_array(precision, "precision")
    recall = _real_array(recall, "recall")
    if recall.ndim != 1 or precision.shape != recall.shape:
        raise ValueError("precision and recall must be aligned vectors")
    if (np.any((precision < 0) | (precision > 1)) or np.any((recall < 0) | (recall > 1))
            or np.any(recall[1:] < recall[:-1])):
        raise ValueError("use precision and nondecreasing recall between zero and one")
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_increases = np.diff(np.concatenate(([0.0], recall)))
    return float(recall_increases @ envelope)
# end-snippet:all_points_ap
