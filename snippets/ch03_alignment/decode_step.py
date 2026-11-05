def step(self, previous, state, memory):
    annotations, projected, source_valid = memory
    context, weights = self.attention(state[0], annotations,
                                      projected, source_valid)
    joined = torch.cat((self.target_embedding(previous), context), dim=-1)
    hidden, cell = self.decoder(joined, state)
    return self.output(hidden), (hidden, cell), weights
