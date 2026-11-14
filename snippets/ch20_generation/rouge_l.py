def lcs_length(candidate, reference):
    """Find a longest common subsequence length with two rows of state."""
    candidate, reference = _tokens(candidate), _tokens(reference)
    previous = [0] * (len(reference) + 1)
    for token in candidate:
        current = [0]
        for column, other in enumerate(reference, start=1):
            if token == other:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(candidate, reference):
    """Return per-example LCS precision, recall, and F1."""
    candidate, reference = _tokens(candidate), _tokens(reference)
    length = lcs_length(candidate, reference)
    total = len(candidate) + len(reference)
    return {
        "lcs_length": length,
        "precision": length / len(candidate) if candidate else 0.0,
        "recall": length / len(reference) if reference else 0.0,
        "f1": 2 * length / total if total else 0.0,
    }
