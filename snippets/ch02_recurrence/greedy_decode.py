def _greedy_steps(model, source, lengths, max_new_tokens):
    state = model.encode(source, lengths)
    previous = torch.full((len(source), 1), 1, device=source.device)  # BOS
    finished = torch.zeros(len(source), dtype=torch.bool, device=source.device)
    generated = []
    for _ in range(max_new_tokens):
        hidden, state = model.decoder(model.target_embedding(previous), state)
        token = model.output(hidden[:, 0]).argmax(dim=-1)
        token = token.masked_fill(finished, 0)  # PAD after an earlier EOS
        generated.append(token)
        finished = finished | (token == 2)  # EOS
        previous = token[:, None]
        if finished.all():
            break
    return torch.stack(generated, dim=1)
