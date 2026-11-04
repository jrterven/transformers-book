"""Arithmetic and a one-layer LSTM encoder-decoder without attention.

The NumPy functions use row vectors. The PyTorch model follows the same
computation with batched token IDs; its hidden and cell states have shape
[1, batch, hidden]. Source lengths, rather than the padded tensor width,
determine the final encoder states passed to the decoder.
"""

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
except ImportError:
    torch = None
    nn = None


# snippet:rnn_step
def rnn_trace(inputs, recurrent_weight=0.5, initial_state=0.0):
    """Reuse one scalar update at every position, returning every state."""
    state = float(initial_state)
    states = []
    for value in inputs:
        state = np.tanh(recurrent_weight * state + float(value))
        states.append(state)
    return np.asarray(states)
# end-snippet:rnn_step


# snippet:lstm_update
def gated_cell_update(cell, forget, write, candidate, expose):
    """The elementwise memory update after the gates have been calculated."""
    new_cell = forget * cell + write * candidate
    new_hidden = expose * np.tanh(new_cell)
    return new_hidden, new_cell
# end-snippet:lstm_update


def lstm_step(x, hidden, cell, weight_input, weight_hidden, bias):
    """One modern LSTM step; gate order is input, forget, candidate, output.

    With input width D and hidden width H, the weights have shapes [D, 4H]
    and [H, 4H], and bias has shape [4H]. Leading batch axes may be present.
    PyTorch stores transposed matrices and two biases; their sum is `bias`.
    """
    gates = np.asarray(x) @ weight_input + hidden @ weight_hidden + bias
    write_raw, forget_raw, candidate_raw, expose_raw = np.split(gates, 4, axis=-1)
    sigmoid = lambda a: np.exp(-np.logaddexp(0.0, -a))
    return gated_cell_update(cell, sigmoid(forget_raw), sigmoid(write_raw),
                             np.tanh(candidate_raw), sigmoid(expose_raw))


if nn is not None:
    class RecurrentSeq2Seq(nn.Module):
        """Separate encoder and decoder LSTMs joined by their final h,c pair."""

        def __init__(self, vocabulary_size=9, embedding_width=16, hidden_width=64):
            super().__init__()
            self.source_embedding = nn.Embedding(vocabulary_size, embedding_width,
                                                 padding_idx=0)
            self.target_embedding = nn.Embedding(vocabulary_size, embedding_width,
                                                 padding_idx=0)
            self.encoder = nn.LSTM(embedding_width, hidden_width, batch_first=True)
            self.decoder = nn.LSTM(embedding_width, hidden_width, batch_first=True)
            self.output = nn.Linear(hidden_width, vocabulary_size)

        # snippet:encode_decode
        def encode(self, source, lengths):
            """Skip right-padding and return the last valid hidden/cell pair."""
            embedded = self.source_embedding(source)
            packed = pack_padded_sequence(embedded, lengths.cpu(),
                                          batch_first=True, enforce_sorted=False)
            _, state = self.encoder(packed)
            return state

        def forward(self, source, lengths, decoder_input):
            state = self.encode(source, lengths)
            embedded = self.target_embedding(decoder_input)
            hidden, _ = self.decoder(embedded, state)
            return self.output(hidden)
        # end-snippet:encode_decode


# snippet:token_loss
def token_loss(logits, target):
    """Average over non-padding labels, including each sequence's EOS."""
    if torch is None:
        raise ImportError("Install requirements-ch02.txt for the PyTorch example")
    if not (target != 0).any():
        raise ValueError("at least one non-padding target is required")
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
        ignore_index=0,
    )
# end-snippet:token_loss


def greedy_decode(model, source, lengths, max_new_tokens):
    """Generate IDs [batch, time], including EOS and padding after early EOS.

    Call model.eval() first. The cap is essential because an untrained model
    may never emit EOS. This routine does not receive the target sequence.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    with torch.no_grad():
        return _greedy_steps(model, source, lengths, max_new_tokens)


# snippet:greedy_decode
def _greedy_steps(model, source, lengths, max_new_tokens):
    state = model.encode(source, lengths)
    previous = torch.full((len(source), 1), 1, device=source.device)  # BOS
    finished = torch.zeros(len(source), dtype=torch.bool, device=source.device)
    generated = []
    for _ in range(max_new_tokens):
        hidden, state = model.decoder(model.target_embedding(previous), state)
        token = model.output(hidden[:, 0]).argmax(dim=-1)
        token = token.masked_fill(finished, 0)  # PAD after an earlier EOS
        generated.append(token)
        finished = finished | (token == 2)  # EOS
        previous = token[:, None]
        if finished.all():
            break
    return torch.stack(generated, dim=1)
# end-snippet:greedy_decode
