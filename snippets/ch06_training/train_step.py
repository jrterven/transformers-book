def train_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch.source, batch.decoder_input)
    total, count = token_loss(logits, batch.target)
    loss = total / count
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise ValueError("non-finite gradient norm")
    optimizer.step()
    return loss.item()
