def span_metrics(gold, predicted, *, repair_predictions=False):
    """Compare labeled spans exactly within one sentence."""
    gold, predicted = tuple(gold), tuple(predicted)
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted tags must refer to the same words")
    gold_spans = bio_spans(gold)
    predicted_spans = bio_spans(predicted, repair=repair_predictions)
    correct = len(gold_spans & predicted_spans)
    n_gold, n_predicted = len(gold_spans), len(predicted_spans)
    total = n_gold + n_predicted
    return {
        "correct_spans": correct,
        "gold_spans": n_gold,
        "predicted_spans": n_predicted,
        "precision": correct / n_predicted if n_predicted else None,
        "recall": correct / n_gold if n_gold else None,
        "f1": 2 * correct / total if total else None,
        "word_accuracy": (
            sum(1 for a, b in zip(gold, predicted) if a == b) / len(gold) if gold else None
        ),
    }
