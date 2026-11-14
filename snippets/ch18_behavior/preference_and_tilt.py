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
