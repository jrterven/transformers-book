def response_loss(logits, labels):
    """Mean negative log-probability over all scored response tokens."""
    return selected_loss(_shifted_logits(logits, labels), labels)


def completion_logprob(logits, labels):
    """Sum selected token log-probabilities separately for each response."""
    shifted = _shifted_logits(logits, labels)
    log_probs = F.log_softmax(shifted, dim=-1)
    selected = labels.ne(IGNORE)
    safe_labels = labels.masked_fill(~selected, 0)
    token_logp = log_probs.gather(-1, safe_labels[..., None]).squeeze(-1)
    return token_logp.masked_fill(~selected, 0.0).sum(dim=-1)
