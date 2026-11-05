def forward(self, ids, start_pos=0):
    if ids.ndim != 2:
        raise ValueError("ids must have shape [batch, tokens]")
    if not isinstance(start_pos, int) or start_pos < 0:
        raise ValueError("start_pos must be a nonnegative integer")
    valid = ids != self.pad_id
    _, allowed = build_masks(valid, valid)
    positions = torch.arange(start_pos, start_pos + ids.shape[1],
                             device=ids.device)
    x = self.embedding(ids)
    for block in self.blocks:
        x = block(x, allowed, positions)
    return self.vocabulary_projection(self.final_norm(x))
