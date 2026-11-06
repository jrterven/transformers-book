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
