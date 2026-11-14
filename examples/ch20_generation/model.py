"""Offline NumPy teaching examples for translation and summarization."""

from collections import Counter
from collections.abc import Mapping
import math
from operator import index

import numpy as np


PAD, BOS, EOS = 0, 1, 2


def _token_id(token):
    if isinstance(token, (bool, np.bool_)):
        raise TypeError("token IDs must be integers")
    token = index(token)
    if not 0 <= token <= np.iinfo(np.int64).max:
        raise ValueError("token IDs must be nonnegative int64 values")
    return token


def _right_pad(rows, pad_id):
    width = max(map(len, rows))
    return np.array([row + [pad_id] * (width - len(row)) for row in rows],
                    dtype=np.int64)


# snippet:collate_pairs
def collate_pairs(pairs, *, pad_id=PAD, bos_id=BOS, eos_id=EOS):
    """Return source IDs, shifted decoder input IDs, and target labels."""
    pad_id, bos_id, eos_id = map(_token_id, (pad_id, bos_id, eos_id))
    special = {pad_id, bos_id, eos_id}
    if len(special) != 3:
        raise ValueError("PAD, BOS, and EOS must have distinct IDs")

    sources, decoder_inputs, labels = [], [], []
    for source, target in pairs:
        source = list(map(_token_id, source))
        target = list(map(_token_id, target))
        if not source or pad_id in source:
            raise ValueError("each source must be nonempty and unpadded")
        if any(token in special for token in target):
            raise ValueError("target content must exclude PAD, BOS, and EOS")
        sources.append(source)
        decoder_inputs.append([bos_id] + target)
        labels.append(target + [eos_id])
    if not sources:
        raise ValueError("a batch needs at least one source-target pair")
    return tuple(_right_pad(rows, pad_id) for rows in (sources, decoder_inputs, labels))
# end-snippet:collate_pairs


def _positive_probabilities(probabilities):
    if not isinstance(probabilities, Mapping):
        raise TypeError("the callback must return a mapping from token IDs to probabilities")
    items = [(_token_id(token), float(probability)) for token, probability in probabilities.items()]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for _, p in items):
        raise ValueError("probabilities must be finite and between zero and one")
    if not math.isclose(math.fsum(p for _, p in items), 1.0, rel_tol=1e-6):
        raise ValueError("callback probabilities must sum to one")
    return [(token, p) for token, p in items if p > 0]


# snippet:beam_search
def beam_search(next_probabilities, beam_width, max_steps, eos_id=EOS):
    """Return the best surviving tokens, log probability, and EOS status."""
    beam_width, max_steps = index(beam_width), index(max_steps)
    eos_id = _token_id(eos_id)
    if min(beam_width, max_steps) < 1:
        raise ValueError("beam width and generation cap must be positive")

    beam = [((), 0.0)]
    for _ in range(max_steps):
        candidates = []
        for prefix, log_probability in beam:
            if prefix and prefix[-1] == eos_id:
                candidates.append((prefix, log_probability))
                continue
            for token, p in _positive_probabilities(next_probabilities(prefix)):
                candidates.append((prefix + (token,), log_probability + math.log(p)))
        # Break equal-score ties by token ID order.
        beam = sorted(candidates, key=lambda item: (-item[1], item[0]))[:beam_width]
        if all(tokens[-1] == eos_id for tokens, _ in beam):
            break
    tokens, log_probability = beam[0]
    return tokens, log_probability, tokens[-1] == eos_id
# end-snippet:beam_search


def _tokens(tokens):
    if isinstance(tokens, (str, bytes)):
        raise TypeError("pass an already tokenized sequence")
    return tuple(tokens)


def _ngram_counts(tokens, n):
    return Counter(tokens[i:i + n] for i in range(len(tokens) - n + 1))


# snippet:clipped_precision
def clipped_ngram_precision(candidate, reference, n):
    """Return clipped precision against one reference, or None for zero trials."""
    n = index(n)
    if n < 1:
        raise ValueError("n must be positive")
    candidate, reference = _tokens(candidate), _tokens(reference)
    count = len(candidate) - n + 1
    if count <= 0:
        return None
    actual = _ngram_counts(candidate, n)
    allowed = _ngram_counts(reference, n)
    matched = sum(min(number, allowed[gram]) for gram, number in actual.items())
    return matched / count
# end-snippet:clipped_precision


# snippet:rouge_l
def lcs_length(candidate, reference):
    """Find a longest common subsequence length with two rows of state."""
    candidate, reference = _tokens(candidate), _tokens(reference)
    previous = [0] * (len(reference) + 1)
    for token in candidate:
        current = [0]
        for column, other in enumerate(reference, start=1):
            if token == other:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(candidate, reference):
    """Return per-example LCS precision, recall, and F1."""
    candidate, reference = _tokens(candidate), _tokens(reference)
    length = lcs_length(candidate, reference)
    total = len(candidate) + len(reference)
    return {
        "lcs_length": length,
        "precision": length / len(candidate) if candidate else 0.0,
        "recall": length / len(reference) if reference else 0.0,
        "f1": 2 * length / total if total else 0.0,
    }
# end-snippet:rouge_l
