def vocabulary_readout_example():
    """A separate three-entry vocabulary: el, coche, EOS.

    The state and readout weights are chosen by hand, not learned.
    """
    state = torch.tensor([[1., -1., 0., 2.]], dtype=torch.float64)
    w_vocab = torch.tensor([
        [1., 0., -1.], [0., -1., 0.], [0., 0., 1.], [.5, 0., .5],
    ], dtype=torch.float64)
    bias = torch.zeros(3, dtype=torch.float64)
    logits = state @ w_vocab + bias
    probabilities = torch.softmax(logits, dim=-1)
    return logits, probabilities
