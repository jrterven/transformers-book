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
