def route_policy(log_scores, min_confidence=None):
    """Choose from A (read), B (write), C (clarify), with optional deferral."""
    chosen = max(range(3), key=log_scores.__getitem__)
    if min_confidence is not None and chosen != 2:
        # Divide by the sum of exponentials above the winning log score.
        probability = 1 / sum(math.exp(score - log_scores[chosen]) for score in log_scores)
        if probability < min_confidence:
            return "C"
    return "ABC"[chosen]
