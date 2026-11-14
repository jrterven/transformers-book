"""Small calculations used to interpret and evaluate an action selection."""

import math


# snippet:patch_effect
def patch_effect(recipient_gap, donor_gap, patched_gap):
    """Return the signed shift and its fraction of the donor-recipient gap."""
    shift = patched_gap - recipient_gap
    span = donor_gap - recipient_gap
    recovery = shift / span if abs(span) > 1e-8 else None
    return shift, recovery
# end-snippet:patch_effect


# snippet:route_policy
def route_policy(log_scores, min_confidence=None):
    """Choose from A (read), B (write), C (clarify), with optional deferral."""
    chosen = max(range(3), key=log_scores.__getitem__)
    if min_confidence is not None and chosen != 2:
        # Divide by the sum of exponentials above the winning log score.
        probability = 1 / sum(math.exp(score - log_scores[chosen]) for score in log_scores)
        if probability < min_confidence:
            return "C"
    return "ABC"[chosen]
# end-snippet:route_policy


def summarize(targets, predictions):
    """Count successes using all cases, including invalid predicted tokens."""
    pairs = list(zip(targets, predictions, strict=True))
    ambiguous = sum(target == "C" for target, _ in pairs)
    return {
        "n": len(pairs),
        "valid": sum(prediction in ("A", "B", "C") for _, prediction in pairs),
        "correct": sum(target == prediction for target, prediction in pairs),
        "right_to_clarify": sum(target == prediction == "C" for target, prediction in pairs),
        "ambiguous": ambiguous,
        "incorrect_write": sum(target != "B" and prediction == "B" for target, prediction in pairs),
    }
