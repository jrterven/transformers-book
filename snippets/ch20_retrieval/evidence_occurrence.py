def literal_evidence_hits(context, phrase):
    """Return IDs that contain the consecutive word tokens of the phrase."""
    wanted = tokenize(phrase)
    if not wanted:
        raise ValueError("the evidence phrase needs at least one word")
    hits = []
    for pid, text in context:
        words = tokenize(text)
        if any(words[i:i + len(wanted)] == wanted
               for i in range(len(words) - len(wanted) + 1)):
            hits.append(pid)
    return tuple(hits)


def evidence_recall(ranked_ids, relevant_ids, k):
    """Return the fraction of supplied gold evidence IDs retrieved by rank k."""
    retrieved = _passage_ids(ranked_ids)
    relevant = set(_passage_ids(relevant_ids))
    k = index(k)
    if k < 0:
        raise ValueError("k must be nonnegative")
    return len(relevant.intersection(retrieved[:k])) / len(relevant) if relevant else None
