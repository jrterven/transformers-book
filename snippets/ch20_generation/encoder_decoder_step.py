def encoder_decoder_step(pairs):
    """Run one supervised update and measure loss on that same batch."""
    arrays = collate_pairs(pairs)
    batch = Batch(*(torch.from_numpy(array) for array in arrays))
    source_vocab = int(batch.source.max()) + 1
    target_vocab = max(int(batch.decoder_input.max()), int(batch.target.max())) + 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = Transformer(
            source_vocab, target_vocab, d_model=8, n_heads=2, d_ff=16,
            dropout=0, max_length=max(batch.source.shape[1], batch.target.shape[1]),
        ).double()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = train_step(model, optimizer, batch)
    model.eval()
    with torch.no_grad():
        logits = model(batch.source, batch.decoder_input)
        total, count = token_loss(logits, batch.target)
    return {
        "logit_shape": list(logits.shape),
        "scored_tokens": int(count),
        "loss_before": before,
        "loss_after": float(total / count),
    }
