"""Additive attention with a small, unidirectional LSTM encoder.

The original Bahdanau model used a bidirectional recurrent encoder and GRUs.
This model keeps Chapter 2's single-direction LSTM so the new retrieval path
can be followed separately. Here context means an attention read; the LSTM
cell state is a different vector.
"""

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ImportError:
    torch = None
    nn = None


# snippet:additive_scores
def additive_scores(query, annotations, weight_query, weight_annotation,
                    direction, bias=0.0):
    """Return scores [..., S] for annotations [..., S, H].

    The shared direction has shape [A] or [A, 1]; both give one scalar
    per source position. Leading query/annotation batch axes may broadcast.
    """
    direction = np.asarray(direction)
    if direction.ndim == 2 and direction.shape[-1] == 1:
        direction = direction[:, 0]
    if direction.ndim != 1:
        raise ValueError("direction must have shape [A] or [A, 1]")
    keys = np.asarray(annotations) @ weight_annotation
    request = np.asarray(query) @ weight_query
    features = np.tanh(keys + np.expand_dims(request, -2) + bias)
    if direction.shape[0] != features.shape[-1]:
        raise ValueError("direction width must match the score-feature width")
    return features @ direction
# end-snippet:additive_scores


def weighted_read(scores, annotations, valid=None):
    """Read annotations [..., S, H] using scores [..., S].

    Source widths must match; leading batch axes may broadcast. Returned
    weights have the scores' shape, while context uses the broadcast batch.
    The optional Boolean validity mask must have the scores' shape.
    """
    scores = np.asarray(scores, dtype=float)
    annotations = np.asarray(annotations)
    if scores.ndim < 1 or annotations.ndim < 2:
        raise ValueError("scores need a source axis; annotations need source/features")
    if scores.shape[-1] != annotations.shape[-2]:
        raise ValueError("provide one score per source annotation")
    try:
        np.broadcast_shapes(scores.shape[:-1], annotations.shape[:-2])
    except ValueError:
        raise ValueError("score and annotation batch axes are incompatible") from None
    valid = np.ones(scores.shape, dtype=bool) if valid is None else np.asarray(valid)
    if valid.dtype != np.bool_ or valid.shape != scores.shape:
        raise ValueError("valid must be a Boolean array with the scores' shape")
    if not valid.any(axis=-1).all():
        raise ValueError("every sequence needs at least one valid source position")
    masked = np.where(valid, scores, -np.inf)
    exponentials = np.exp(masked - masked.max(axis=-1, keepdims=True))
    weights = exponentials / exponentials.sum(axis=-1, keepdims=True)
    context = (weights[..., None] * annotations).sum(axis=-2)
    return weights, context


if nn is not None:
    class AdditiveAttention(nn.Module):
        """Compare a decoder query with every available source annotation."""

        def __init__(self, hidden_width=64, alignment_width=32):
            super().__init__()
            self.annotation = nn.Linear(hidden_width, alignment_width, bias=False)
            self.query = nn.Linear(hidden_width, alignment_width, bias=False)
            self.bias = nn.Parameter(torch.zeros(alignment_width))
            self.energy = nn.Linear(alignment_width, 1, bias=False)

        # snippet:masked_context
        def forward(self, query, annotations, projected, source_valid):
            if not source_valid.any(dim=1).all():
                raise ValueError("every source needs at least one valid position")
            features = projected + self.query(query)[:, None, :] + self.bias
            scores = self.energy(torch.tanh(features)).squeeze(-1)
            scores = scores.masked_fill(~source_valid, -torch.inf)
            weights = torch.softmax(scores, dim=-1)
            context = torch.bmm(weights[:, None, :], annotations).squeeze(1)
            return context, weights
        # end-snippet:masked_context


    class AttentiveSeq2Seq(nn.Module):
        """One encoder LSTM and one decoder LSTMCell with additive retrieval."""

        def __init__(self, vocabulary_size=9, embedding_width=16,
                     hidden_width=64, alignment_width=32):
            super().__init__()
            self.source_embedding = nn.Embedding(vocabulary_size, embedding_width,
                                                 padding_idx=0)
            self.target_embedding = nn.Embedding(vocabulary_size, embedding_width,
                                                 padding_idx=0)
            self.encoder = nn.LSTM(embedding_width, hidden_width, batch_first=True)
            self.attention = AdditiveAttention(hidden_width, alignment_width)
            self.decoder = nn.LSTMCell(embedding_width + hidden_width, hidden_width)
            self.output = nn.Linear(hidden_width, vocabulary_size)

        # snippet:encoder_annotations
        def encode(self, source, lengths):
            if (source.ndim != 2 or lengths.shape != (len(source),)
                    or len(source) == 0 or (lengths < 1).any()
                    or (lengths > source.shape[1]).any()):
                raise ValueError("provide a nonempty source batch and valid lengths")
            valid = torch.arange(source.shape[1], device=source.device)[None, :]
            valid = valid < lengths.to(source.device)[:, None]
            if (source[valid] == 0).any():
                raise ValueError("valid source positions must not be PAD")
            packed = pack_padded_sequence(self.source_embedding(source), lengths.cpu(),
                                          batch_first=True, enforce_sorted=False)
            packed_states, (hidden, cell) = self.encoder(packed)
            annotations, _ = pad_packed_sequence(
                packed_states, batch_first=True, total_length=source.shape[1])
            projected = self.attention.annotation(annotations)
            return (annotations, projected, valid), (hidden[0], cell[0])
        # end-snippet:encoder_annotations

        # snippet:decode_step
        def step(self, previous, state, memory):
            annotations, projected, source_valid = memory
            context, weights = self.attention(state[0], annotations,
                                              projected, source_valid)
            joined = torch.cat((self.target_embedding(previous), context), dim=-1)
            hidden, cell = self.decoder(joined, state)
            return self.output(hidden), (hidden, cell), weights
        # end-snippet:decode_step

        # snippet:teacher_forward
        def forward(self, source, lengths, decoder_input):
            memory, state = self.encode(source, lengths)
            if decoder_input.ndim != 2 or decoder_input.shape[1] == 0:
                raise ValueError("decoder inputs need at least one position")
            logits, alignments = [], []
            for previous in decoder_input.unbind(dim=1):
                scores, state, weights = self.step(previous, state, memory)
                logits.append(scores)
                alignments.append(weights)
            return torch.stack(logits, dim=1), torch.stack(alignments, dim=1)
        # end-snippet:teacher_forward


def greedy_decode(model, source, lengths, max_new_tokens):
    """Return generated IDs and their attention rows.

    Call model.eval() first. Each row includes attention used to emit EOS.
    Positions after an earlier EOS are PAD with zero attention weights.
    A pre-EOS PAD or BOS selection is an error, not a stopping condition.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    with torch.no_grad():
        return _greedy_steps(model, source, lengths, max_new_tokens)


# snippet:greedy_decode
def _greedy_steps(model, source, lengths, max_new_tokens):
    memory, state = model.encode(source, lengths)
    previous = torch.full((len(source),), 1, device=source.device)  # BOS
    finished = torch.zeros(len(source), dtype=torch.bool, device=source.device)
    generated, alignments = [], []
    for _ in range(max_new_tokens):
        logits, state, weights = model.step(previous, state, memory)
        token = logits.argmax(dim=-1).masked_fill(finished, 0)
        weights = weights.masked_fill(finished[:, None], 0.0)
        generated.append(token)
        alignments.append(weights)
        finished = finished | (token == 2)  # EOS
        previous = token
        if finished.all():
            break
    return torch.stack(generated, dim=1), torch.stack(alignments, dim=1)
# end-snippet:greedy_decode
