def word_error_rate(reference, hypothesis):
    """Find minimum edit counts and divide by the reference word count."""
    reference, hypothesis = map(_words, (reference, hypothesis))
    # Store (substitutions, deletions, insertions) for each prefix pair.
    previous = [(0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, actual in enumerate(reference, start=1):
        current = [(0, i, 0)]
        for j, guess in enumerate(hypothesis, start=1):
            s, d, ins = previous[j - 1]
            diagonal = (s + int(actual != guess), d, ins)
            s, d, ins = previous[j]
            deletion = (s, d + 1, ins)
            s, d, ins = current[-1]
            insertion = (s, d, ins + 1)
            # Ties prefer the diagonal, then deletion, then insertion.
            current.append(min((diagonal, deletion, insertion), key=sum))
        previous = current

    substitutions, deletions, insertions = previous[-1]
    edits = substitutions + deletions + insertions
    return {
        "substitutions": substitutions, "deletions": deletions,
        "insertions": insertions, "edits": edits,
        "reference_words": len(reference),
        "wer": edits / len(reference) if reference else None,
    }
