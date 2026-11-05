def bert_corrupt(tokens, selected, generator=None):
    original = _checked_tokens(tokens)
    selected = _checked_selected(original, selected)
    inputs = original.clone()
    labels = torch.full_like(original, IGNORE)
    for i in selected:
        labels[i] = original[i]
        draw = torch.rand((), generator=generator).item()
        if draw < 0.8:
            inputs[i] = MASK
        elif draw < 0.9:
            inputs[i] = torch.randint(
                FIRST_WORD, len(VOCAB), (), generator=generator
            )
        # Otherwise keep the original input, but still score its label.
    return inputs, labels
