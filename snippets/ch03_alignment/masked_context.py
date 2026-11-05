def forward(self, query, annotations, projected, source_valid):
    if not source_valid.any(dim=1).all():
        raise ValueError("every source needs at least one valid position")
    features = projected + self.query(query)[:, None, :] + self.bias
    scores = self.energy(torch.tanh(features)).squeeze(-1)
    scores = scores.masked_fill(~source_valid, -torch.inf)
    weights = torch.softmax(scores, dim=-1)
    context = torch.bmm(weights[:, None, :], annotations).squeeze(1)
    return context, weights
