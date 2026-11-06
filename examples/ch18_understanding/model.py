"""NumPy teaching examples for sentence classification and token labeling."""

from operator import index

import numpy as np


IGNORE = -100


def _real_array(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    return values.astype(np.float64, copy=False)


# snippet:sentence_logits
def sentence_logits(features, valid, weight, bias):
    """Return masked means and logits for one sentence or a batch."""
    features = _real_array(features, "features")
    weight = _real_array(weight, "weight")
    bias = _real_array(bias, "bias")
    valid = np.asarray(valid)
    if features.ndim not in (2, 3) or min(features.shape) == 0:
        raise ValueError("features need shape [time, width] or [batch, time, width]")
    if valid.shape != features.shape[:-1] or valid.dtype != np.bool_:
        raise ValueError("valid must be a boolean mask matching the token axes")
    if (weight.ndim != 2 or min(weight.shape) == 0
            or weight.shape[1] != features.shape[-1] or bias.shape != weight.shape[:1]):
        raise ValueError("weight needs [classes, width] and bias needs [classes]")
    count = valid.sum(axis=-1)
    if np.any(count == 0):
        raise ValueError("each sentence must select at least one token")
    if not all(np.isfinite(x).all() for x in (features[valid], weight, bias)):
        raise ValueError("selected features and head parameters must be finite")

    with np.errstate(over="raise", invalid="raise"):
        kept = np.where(valid[..., None], features, 0)
        pooled = kept.sum(axis=-2) / count[..., None]
        logits = pooled @ weight.T + bias
    return pooled, logits
# end-snippet:sentence_logits


def _integer_vector(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a vector of integers")
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must fit int64")
    return values.astype(np.int64, copy=False)


def _first_piece_positions(word_ids):
    ids = _integer_vector(word_ids, "word IDs")
    if np.any(ids < -1):
        raise ValueError("use -1 for special or padding tokens")
    previous = np.concatenate(([-1], ids))[:-1]
    first = np.flatnonzero((ids >= 0) & (ids != previous))
    if not np.array_equal(ids[first], np.arange(len(first))):
        raise ValueError("word IDs must be consecutive from zero, with contiguous pieces")
    return ids, first


# snippet:first_piece_labels
def first_piece_labels(word_ids, word_labels, ignore=IGNORE):
    """Place one supervised tag at the first piece of each word."""
    ids, first = _first_piece_positions(word_ids)
    labels = _integer_vector(word_labels, "word labels")
    ignore = index(ignore)
    if len(labels) != len(first) or np.any(labels < 0):
        raise ValueError("each word needs one nonnegative tag")
    if not np.iinfo(np.int64).min <= ignore < 0:
        raise ValueError("the ignore label must be a negative int64 value")
    aligned = np.full(len(ids), ignore, dtype=np.int64)
    aligned[first] = labels
    return aligned


def first_piece_predictions(word_ids, token_predictions):
    """Recover word predictions from the same first-piece positions."""
    ids, first = _first_piece_positions(word_ids)
    predictions = _integer_vector(token_predictions, "token predictions")
    if predictions.shape != ids.shape or np.any(predictions[first] < 0):
        raise ValueError("each token needs a prediction, with valid tags at first pieces")
    return predictions[first]
# end-snippet:first_piece_labels


# snippet:bio_spans
def bio_spans(tags, *, repair=False):
    """Read BIO spans as (start, exclusive end, entity type) triples."""
    spans = set()
    start, entity = 0, None
    # A final O closes a span that reaches the end of the sentence.
    for position, tag in enumerate((*tags, "O")):
        if not isinstance(tag, str):
            raise ValueError("BIO tags must be strings")
        prefix, _, label = tag.partition("-")
        if tag != "O":
            if prefix not in ("B", "I") or not label:
                raise ValueError(f"invalid BIO tag at word {position}: {tag}")
            if prefix == "I":
                if label == entity:
                    continue
                if not repair:
                    raise ValueError(f"unexpected inside tag at word {position}: {tag}")

        if entity is not None:
            spans.add((start, position, entity))
        start = position
        entity = None if tag == "O" else label
    return spans
# end-snippet:bio_spans


# snippet:span_metrics
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
# end-snippet:span_metrics
