model.eval()
loss_sum = token_count = correct_tokens = correct_sequences = terminated = 0
for start in range(0, len(sequences), batch_size):
    batch = collate(sequences[start:start + batch_size])
    logits = model(batch.source, batch.decoder_input)
    total, count = token_loss(logits, batch.target)
    valid = batch.target != PAD
    loss_sum += total.item()
    token_count += count.item()
    correct_tokens += ((logits.argmax(-1) == batch.target) & valid).sum().item()
    generated = greedy_decode(model, batch.source,
                              max_new_tokens=batch.target.shape[1] + 1)
    correct_sequences += exact_matches(generated, batch.target)
    terminated += generated.eq(EOS).any(dim=1).sum().item()
nll = loss_sum / token_count
return {"nll": nll, "perplexity": math.exp(nll),
        "token_accuracy": correct_tokens / token_count,
        "exact_match": correct_sequences / len(sequences),
        "termination_rate": terminated / len(sequences),
        "tokens": token_count, "sequences": len(sequences)}
