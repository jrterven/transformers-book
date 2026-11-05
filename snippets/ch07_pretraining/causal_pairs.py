def causal_pairs(tokens):
    tokens = _checked_tokens(tokens)
    if len(tokens) < 2:
        raise ValueError("next-token prediction needs at least two tokens")
    inputs = tokens[:-1]
    labels = tokens[1:]
    return inputs, labels
