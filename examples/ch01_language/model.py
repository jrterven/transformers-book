"""A hand-chosen, two-token-window neural language model in NumPy.

These weights illustrate a forward pass. They were not trained on text and do
not measure language ability. The last two token embeddings are concatenated
in order, passed through an affine map and ReLU, then projected to all sixteen
vocabulary scores. Arrays use row-vector multiplication.
"""

import numpy as np


VOCAB = ("<pad>", "<bos>", "<eos>", "blue", "the", "green", "bike", "car",
         "truck", "a", "is", "fast", "red", "slow", "saw", "I")
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
PAD, BOS, EOS = 0, 1, 2


def _check_ids(ids, vocabulary_size):
    ids = np.asarray(ids)
    if ids.dtype.kind not in "iu":
        raise TypeError("token IDs must be integers")
    if np.any(ids < 0) or np.any(ids >= vocabulary_size):
        raise ValueError("token ID is outside the vocabulary")
    return ids


# snippet:embedding_lookup
def embedding_lookup(embedding_table, token_ids):
    """Append the embedding-width axis to the token-ID array's shape."""
    embedding_table = np.asarray(embedding_table, dtype=np.float64)
    if embedding_table.ndim != 2 or min(embedding_table.shape) < 1:
        raise ValueError("embedding_table must be a nonempty [V, d] matrix")
    token_ids = _check_ids(token_ids, embedding_table.shape[0])
    return embedding_table[token_ids]
# end-snippet:embedding_lookup


# snippet:stable_softmax
def stable_softmax(logits):
    """Normalize each vocabulary row independently along its last axis."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim < 1 or logits.shape[-1] == 0 or not np.isfinite(logits).all():
        raise ValueError("logits must have a nonempty, finite vocabulary axis")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)
# end-snippet:stable_softmax


# snippet:context_model
def context_logits(context_ids, parameters):
    """Map [..., 2] token IDs to [..., 4] hidden vectors and [..., 16] logits."""
    x = embedding_lookup(parameters["E"], context_ids)
    if x.ndim < 2 or x.shape[-2] != 2:
        raise ValueError("each context must contain exactly two token IDs")
    joined = x.reshape(x.shape[:-2] + (2 * x.shape[-1],))
    hidden = np.maximum(0.0, joined @ parameters["W_context"]
                        + parameters["b_context"])
    logits = hidden @ parameters["W_out"] + parameters["b_out"]
    return hidden, logits
# end-snippet:context_model


# snippet:shifted_pairs
def next_token_pairs(token_ids):
    """Use two BOS tokens to fill missing context at the start of a sentence."""
    if len(token_ids) == 0:
        raise ValueError("a sentence must contain at least one word")
    token_ids = _check_ids(token_ids, len(VOCAB))
    if token_ids.ndim != 1 or np.isin(token_ids, (PAD, BOS, EOS)).any():
        raise ValueError("sentence IDs must be a one-dimensional list of word IDs")
    padded = np.concatenate(([BOS, BOS], token_ids, [EOS]))
    contexts = np.stack([padded[t:t + 2] for t in range(len(token_ids) + 1)])
    targets = padded[2:]
    return contexts, targets
# end-snippet:shifted_pairs


def token_nll(logits, targets):
    """Negative log-probability of each target, without rounding probabilities."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim < 1 or logits.shape[-1] == 0 or not np.isfinite(logits).all():
        raise ValueError("logits must have a nonempty, finite vocabulary axis")
    targets = _check_ids(targets, logits.shape[-1])
    if targets.shape != logits.shape[:-1]:
        raise ValueError("one target ID is required for each vocabulary row")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return -np.take_along_axis(log_probabilities, targets[..., None], axis=-1)[..., 0]


def toy_parameters():
    """Return the fixed teaching matrices; all unshown embedding rows are zero."""
    embedding_table = np.zeros((len(VOCAB), 4), dtype=np.float64)
    embedding_table[TOKEN_TO_ID["the"]] = (0.2, -0.4, 0.1, 0.5)
    embedding_table[TOKEN_TO_ID["red"]] = (-0.3, 0.8, 0.2, -0.1)
    embedding_table[TOKEN_TO_ID["car"]] = (0.6, 0.1, -0.5, 0.3)
    output_weight = np.zeros((4, len(VOCAB)))
    output_weight[1, TOKEN_TO_ID["car"]] = 2.0
    output_weight[1, TOKEN_TO_ID["red"]] = 1.0
    return {"E": embedding_table,
            "W_context": np.vstack((np.eye(4), 2 * np.eye(4))),
            "b_context": np.array((0.0, -0.2, 0.0, 0.0)),
            "W_out": output_weight, "b_out": np.zeros(len(VOCAB))}
