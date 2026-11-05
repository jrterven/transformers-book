def span_corrupt(tokens, spans):
    tokens = _checked_tokens(tokens).tolist()
    spans = _checked_spans(torch.tensor(tokens), spans)
    source, target, cursor = [], [], 0
    for number, (start, end) in enumerate(spans):
        sentinel = SENTINELS[number]
        source.extend(tokens[cursor:start] + [sentinel])
        target.extend([sentinel] + tokens[start:end])
        cursor = end
    source.extend(tokens[cursor:] + [EOS])
    target.extend([SENTINELS[len(spans)], EOS])
    decoder_input = [BOS] + target[:-1]
    return tuple(torch.tensor(row) for row in
                 (source, decoder_input, target))
