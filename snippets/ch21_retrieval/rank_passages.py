def rank_passages(passage_ids, scores, limit=None):
    """Rank higher scores first and break ties by ascending passage ID."""
    ids = _passage_ids(passage_ids)
    scores = np.asarray(scores)
    if (scores.shape != (len(ids),) or scores.dtype.kind not in "iuf"
            or not np.isfinite(scores).all()):
        raise ValueError("supply one finite real score per passage ID")
    if limit is not None:
        limit = index(limit)
        if limit < 0:
            raise ValueError("the rank limit must be nonnegative")
    by_id = sorted(zip(ids, scores))
    ranked = sorted(by_id, key=lambda pair: pair[1], reverse=True)
    return tuple(pid for pid, _ in ranked[:limit])


def rerank_candidates(candidate_ids, paired_scores):
    """Order only the candidates using scores keyed by their IDs."""
    ids = _passage_ids(candidate_ids)
    if not isinstance(paired_scores, Mapping) or set(paired_scores) != set(ids):
        raise ValueError("supply exactly one paired score per candidate ID")
    return rank_passages(ids, [paired_scores[pid] for pid in ids])
