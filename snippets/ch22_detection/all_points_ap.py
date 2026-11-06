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
