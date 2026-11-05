def loss_sanity_example():
    """Uniform next-token probabilities in a two-example padded batch."""
    batch = collate([(3, 4, 5), (3, 5)])
    logits = torch.zeros(*batch.target.shape, len(VOCAB), dtype=torch.float64)
    total, count = token_loss(logits, batch.target)
    nll = total / count
    return {"loss_sum": total.item(), "tokens": count.item(),
            "nll": nll.item(), "perplexity": torch.exp(nll).item()}
