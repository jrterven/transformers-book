"""Build three teaching tasks from text; no tokenizer or dataset download."""

import torch


VOCAB = ("PAD", "BOS", "EOS", "[MASK]", "S0", "S1", "S2",
         "the", "red", "car", "stopped", "near", "old", "bridge", "today",
         "blue", "moved")
PAD, BOS, EOS, MASK = range(4)
SENTINELS = (4, 5, 6)
IGNORE = -100
FIRST_WORD = 7
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}


def encode(text):
    """One whitespace-separated word is one token in this small vocabulary."""
    return [TOKEN_TO_ID[word] for word in text.split()]


def token_names(ids):
    return ["-" if int(token) == IGNORE else VOCAB[int(token)] for token in ids]


def _checked_tokens(tokens):
    result = torch.as_tensor(tokens)
    if result.is_floating_point() or result.is_complex() or result.dtype == torch.bool:
        raise TypeError("token IDs must be integers")
    result = result.to(dtype=torch.long).clone()
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError("tokens must be a nonempty one-dimensional sequence")
    if ((result < 0) | (result >= len(VOCAB))).any():
        raise ValueError("token ID is outside the teaching vocabulary")
    return result


# snippet:causal_pairs
def causal_pairs(tokens):
    tokens = _checked_tokens(tokens)
    if len(tokens) < 2:
        raise ValueError("next-token prediction needs at least two tokens")
    inputs = tokens[:-1]
    labels = tokens[1:]
    return inputs, labels
# end-snippet:causal_pairs


def _checked_selected(tokens, selected):
    selected = tuple(selected)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("choose one or more distinct prediction positions")
    if any(not isinstance(i, int) or i < 0 or i >= len(tokens) for i in selected):
        raise ValueError("prediction position is outside the sequence")
    if any(tokens[i] < FIRST_WORD for i in selected):
        raise ValueError("select ordinary words, not special markers")
    return selected


# snippet:bert_corruption
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
# end-snippet:bert_corruption


def _checked_spans(tokens, spans):
    spans = tuple(tuple(span) for span in spans)
    # One extra sentinel closes the paper-style target illustrated here.
    if not spans or len(spans) >= len(SENTINELS):
        raise ValueError("choose one or two spans for this small vocabulary")
    previous_end = -1
    for start, end in spans:
        if not (isinstance(start, int) and isinstance(end, int)
                and 0 <= start < end <= len(tokens)):
            raise ValueError("each span must be a nonempty half-open interval")
        if start <= previous_end:
            raise ValueError("spans must be ordered and separated by visible text")
        if (tokens[start:end] < FIRST_WORD).any():
            raise ValueError("remove ordinary words, not special markers")
        previous_end = end
    return spans


# snippet:span_corruption
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
# end-snippet:span_corruption
