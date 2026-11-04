"""A deterministic, disjoint reversal dataset with a fixed vocabulary."""

from dataclasses import dataclass
import random

import torch


VOCAB = ("<pad>", "<bos>", "<eos>", "red", "green", "blue",
         "yellow", "white", "black")
PAD, BOS, EOS = 0, 1, 2
COLORS = tuple(range(3, len(VOCAB)))


def make_splits(seed=2026, n_train=512, n_validation=64, n_test=64,
                min_length=3, max_length=6):
    """Sample unique sequences; give every length the same split counts.

    Counts must be divisible by the number of lengths. Repeated colors within
    a sequence are allowed. No exact source sequence appears in two splits.
    """
    sizes = (n_train, n_validation, n_test)
    lengths = tuple(range(min_length, max_length + 1))
    if not lengths or min_length < 1 or any(n < 1 or n % len(lengths) for n in sizes):
        raise ValueError("positive split sizes must divide evenly across lengths")
    counts = tuple(n // len(lengths) for n in sizes)
    rng = random.Random(seed)
    result = [[], [], []]
    for length in lengths:
        population_size = len(COLORS) ** length
        if sum(counts) > population_size:
            raise ValueError("requested more unique sequences than this length has")
        # Sample positions in the possible-sequence ordering, not every tuple.
        # This keeps the seeded examples unchanged without allocating 6**length
        # tuples when trying the longer-sequence exercise.
        indices = rng.sample(range(population_size), sum(counts))
        chosen = []
        for index in indices:
            sequence = [COLORS[0]] * length
            for position in range(length - 1, -1, -1):
                index, color_index = divmod(index, len(COLORS))
                sequence[position] = COLORS[color_index]
            chosen.append(tuple(sequence))
        start = 0
        for split, count in zip(result, counts):
            split.extend(chosen[start:start + count])
            start += count
    for split in result:
        rng.shuffle(split)
    return tuple(result)


@dataclass
class Batch:
    source: torch.Tensor
    decoder_input: torch.Tensor
    target: torch.Tensor

    @property
    def source_valid(self):
        return self.source != PAD

    @property
    def target_valid(self):
        return self.decoder_input != PAD


def _pad(rows):
    width = max(map(len, rows))
    return torch.tensor([list(row) + [PAD] * (width - len(row)) for row in rows])


# snippet:data_shift
def collate(sequences):
    source = [list(sequence) for sequence in sequences]
    if not source or any(not row or any(t not in COLORS for t in row)
                         for row in source):
        raise ValueError("each source must be a nonempty sequence of color IDs")
    targets = [list(reversed(row)) for row in source]
    decoder_input = [[BOS] + row for row in targets]
    labels = [row + [EOS] for row in targets]
    return Batch(_pad(source), _pad(decoder_input), _pad(labels))
# end-snippet:data_shift


def words(ids, include_special=False):
    """Render IDs, stopping after EOS; include EOS only when requested."""
    result = []
    for token in ids:
        token = int(token)
        if token == EOS:
            if include_special:
                result.append(VOCAB[token])
            break
        if include_special or token in COLORS:
            result.append(VOCAB[token])
    return " ".join(result)
