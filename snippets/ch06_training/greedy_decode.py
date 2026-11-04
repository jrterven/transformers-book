batch_size = source.shape[0]
prefix = source.new_full((batch_size, 1), BOS)
finished = torch.zeros(batch_size, dtype=torch.bool, device=source.device)
generated = []
for _ in range(max_new_tokens):
    logits = (cache.step(prefix[:, -1:]) if cached else
              decode_prefix(model, prefix, memory, source_allowed)[:, -1])
    logits[:, [PAD, BOS]] = float("-inf")
    selected = logits.argmax(dim=-1)
    # Keep finished rows inert. EOS remains a valid cache input.
    selected = torch.where(finished, EOS, selected)
    generated.append(torch.where(finished, PAD, selected))
    finished |= selected == EOS
    prefix = torch.cat((prefix, selected[:, None]), dim=1)
    if finished.all():
        break
return torch.stack(generated, dim=1)
