"""Finite-action versions of the reward and policy calculations in Chapter 18."""

import math


# snippet:preference_and_tilt
def preference_probability(score_left, score_right):
    gap = score_left - score_right
    term = math.exp(-abs(gap))
    return 1.0 / (1.0 + term) if gap >= 0 else term / (1.0 + term)


def reward_tilt(log_reference, reward, beta):
    """Exact KL-regularized optimum over a finite set of actions."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    scores = [log_p + r / beta for log_p, r in zip(log_reference, reward, strict=True)]
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    return [w / sum(weights) for w in weights]
# end-snippet:preference_and_tilt


def expected_cost(p_intent, loss):
    """One cost per proposed action, averaging over possible true intents."""
    return [math.fsum(p*c for p,c in zip(p_intent, row, strict=True)) for row in loss]


def reward_curves(record, strengths=(0, 0.25, 0.5, 1, 2, 4, 8)):
    """Rescore recorded Qwen actions with an immediate-resolution reward."""
    rows = [row for split in record["routing"].values() for row in split["cases"]]
    result = []
    for strength in strengths:
        probs = [reward_tilt(row["candidate_logits"], (strength, strength, 0), beta=1)
                 for row in rows]
        correct, clarify, acted, unsafe = 0.0, 0.0, 0.0, 0.0
        ambiguous = sum(row["target"] == "C" for row in rows)
        for row, p in zip(rows, probs, strict=True):
            target = row["target"]
            correct += p["ABC".index(target)]
            acted += p[0] + p[1]
            if target == "C":
                clarify += p[2]
            if target != "B":
                unsafe += p[1]
        result.append({"strength": strength, "immediate_action": acted / len(rows),
                       "correct": correct / len(rows), "clarify_if_ambiguous": clarify / ambiguous,
                       "incorrect_updates": unsafe / len(rows)})
    return result
