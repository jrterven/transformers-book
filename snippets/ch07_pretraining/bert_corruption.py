def bert_corrupt(tokens, selected, generator=None, *, return_trace=False):
    original = _checked_tokens(tokens)
    selected = _checked_selected(original, selected)
    inputs = original.clone()
    labels = torch.full_like(original, IGNORE)
    trace = []
    for i in selected:
        labels[i] = original[i]
        draw = torch.rand((), generator=generator).item()
        if draw < 0.8:
            inputs[i] = MASK
            treatment = "mask"
        elif draw < 0.9:
            inputs[i] = torch.randint(
                FIRST_WORD, len(VOCAB), (), generator=generator
            )
            treatment = "random"
        else:
            treatment = "unchanged"  # Still score the original label.
        if return_trace:
            trace.append(dict(position=i, draw=draw, treatment=treatment,
                              input_id=int(inputs[i]), label_id=int(labels[i])))
    return (inputs, labels, trace) if return_trace else (inputs, labels)
