"""A balanced, disjoint synthetic color-reversal dataset."""

from dataclasses import dataclass
import random

import torch


VOCAB = ("<pad>", "<bos>", "<eos>", "red", "green", "blue",
         "yellow", "white", "black")
PAD, BOS, EOS = 0, 1, 2
COLORS = tuple(range(3, len(VOCAB)))


def make_splits(seed=2026, sizes=(192, 48, 48), lengths=(3, 4, 5, 6)):
    """Use unique source tuples, with the same count at every source length."""
    if (not lengths or min(lengths) < 1 or len(set(lengths)) != len(lengths)
            or len(sizes) != 3 or any(n < 1 or n % len(lengths) for n in sizes)):
        raise ValueError("provide three positive split sizes balanced over lengths")
    rng = random.Random(seed)
    splits = [[], [], []]
    counts = [n // len(lengths) for n in sizes]
    for length in lengths:
        if sum(counts) > len(COLORS) ** length:
            raise ValueError("not enough distinct source sequences")
        indices = rng.sample(range(len(COLORS) ** length), sum(counts))
        sequences = []
        for index in indices:
            row = [COLORS[0]] * length
            for j in range(length - 1, -1, -1):
                index, color = divmod(index, len(COLORS))
                row[j] = COLORS[color]
            sequences.append(tuple(row))
        start = 0
        for split, count in zip(splits, counts):
            split.extend(sequences[start:start + count])
            start += count
    for split in splits:
        rng.shuffle(split)
    return tuple(splits)


@dataclass
class Batch:
    source: torch.Tensor
    lengths: torch.Tensor
    decoder_input: torch.Tensor
    target: torch.Tensor


def pad(rows):
    width = max(map(len, rows))
    return torch.tensor([list(row) + [PAD] * (width - len(row)) for row in rows])


# snippet:teacher_forcing
def collate(sequences):
    """BOS goes into the decoder; EOS is one of the labels it must predict."""
    source = [list(row) for row in sequences]
    if not source or any(not row or any(t not in COLORS for t in row)
                         for row in source):
        raise ValueError("sources must be nonempty lists of color IDs")
    reversed_rows = [list(reversed(row)) for row in source]
    decoder_input = [[BOS] + row for row in reversed_rows]
    target = [row + [EOS] for row in reversed_rows]
    return Batch(pad(source), torch.tensor([len(row) for row in source]),
                 pad(decoder_input), pad(target))
# end-snippet:teacher_forcing


def words(ids):
    """Display generated tokens, retaining special tokens before the first EOS."""
    result = []
    for token in ids:
        result.append(VOCAB[int(token)])
        if int(token) == EOS:
            break
    return " ".join(result)
