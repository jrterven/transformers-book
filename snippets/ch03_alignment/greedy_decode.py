def _greedy_steps(model, source, lengths, max_new_tokens):
    memory, state = model.encode(source, lengths)
    previous = torch.full((len(source),), 1, device=source.device)  # BOS
    finished = torch.zeros(len(source), dtype=torch.bool, device=source.device)
    generated, alignments = [], []
    for _ in range(max_new_tokens):
        logits, state, weights = model.step(previous, state, memory)
        token = logits.argmax(dim=-1).masked_fill(finished, 0)
        weights = weights.masked_fill(finished[:, None], 0.0)
        generated.append(token)
        alignments.append(weights)
        finished = finished | (token == 2)  # EOS
        previous = token
        if finished.all():
            break
    return torch.stack(generated, dim=1), torch.stack(alignments, dim=1)
