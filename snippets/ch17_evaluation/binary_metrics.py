def binary_metrics(y_true, y_pred):
    """Count the four outcomes and compute metrics on aligned rows."""
    truth, pred = _aligned_predictions(y_true, y_pred)
    tp = int(np.count_nonzero(truth & pred))
    fn = int(np.count_nonzero(truth & ~pred))
    fp = int(np.count_nonzero(~truth & pred))
    tn = int(np.count_nonzero(~truth & ~pred))

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": (tp + tn) / len(truth),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }
