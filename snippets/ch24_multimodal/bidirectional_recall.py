def _recall_at_k(scores, relevance, k):
    per_query = []
    for row, relevant in zip(scores, relevance):
        total = int(relevant.sum())
        if total == 0:
            per_query.append(None)
            continue
        # Stable sorting keeps candidate order for ties.
        best = sorted(range(len(row)), key=row.__getitem__, reverse=True)[:k]
        per_query.append(int(relevant[best].sum()) / total)
    defined = [value for value in per_query if value is not None]
    return {
        "per_query": per_query,
        "defined_queries": len(defined),
        "mean": float(np.mean(defined)) if defined else None,
    }


def bidirectional_recall_at_k(scores, relevance, k):
    """Average the fraction of each query's relevant items in its top k."""
    scores = _real_matrix(scores, "scores", allow_empty=True)
    relevance = np.asarray(relevance)
    if (relevance.shape != scores.shape or relevance.dtype.kind not in "biu"
            or np.any((relevance != 0) & (relevance != 1))):
        raise ValueError("relevance must be a binary matrix matching the scores")
    k = index(k)
    if k < 1:
        raise ValueError("k must be positive")
    return {
        "image_to_text": _recall_at_k(scores, relevance, k),
        "text_to_image": _recall_at_k(scores.T, relevance.T, k),
    }
