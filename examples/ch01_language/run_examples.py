"""Print Chapter 1's hand-calculated examples; no training or downloads."""

import numpy as np

from .model import (VOCAB, TOKEN_TO_ID, context_logits, embedding_lookup,
                    next_token_pairs, stable_softmax, token_nll, toy_parameters)


def main():
    np.set_printoptions(precision=6, suppress=True)
    parameters = toy_parameters()
    # snippet:example
    sentence_ids = np.array([TOKEN_TO_ID[word] for word in ("the", "red", "car")])
    x = embedding_lookup(parameters["E"], sentence_ids)
    hidden, logits = context_logits(sentence_ids[:2], parameters)
    probabilities = stable_softmax(logits)
    best_id = int(probabilities.argmax())
    print("Token IDs:", sentence_ids)
    print("Embedding matrix X:\n", x)
    print("Context hidden vector:", hidden)
    print("All vocabulary probabilities:", probabilities)
    print("Most probable next token:", VOCAB[best_id])
    print("Loss for target 'car':", token_nll(logits, TOKEN_TO_ID["car"]))
    # end-snippet:example
    print("\nHand-chosen parameters, not a trained language model.")
    for token in ("car", "red", "the"):
        print(f"P({token} | the red) = {probabilities[TOKEN_TO_ID[token]]:.12f}")
    reversed_hidden, reversed_logits = context_logits(sentence_ids[:2][::-1], parameters)
    print("Hidden vector for reversed context:", reversed_hidden)
    print("Reversed-context probabilities:", stable_softmax(reversed_logits))
    batch_ids = np.stack((sentence_ids, sentence_ids[::-1]))
    print("Batch IDs / embeddings:", batch_ids.shape,
          embedding_lookup(parameters["E"], batch_ids).shape)
    contexts, targets = next_token_pairs(sentence_ids)
    print("Next-token training pairs (last two inputs -> label):")
    for row, target in zip(contexts, targets):
        print(" ", " ".join(VOCAB[i] for i in row), "->", VOCAB[target])
    np.testing.assert_allclose(hidden, (0.0, 1.0, 0.5, 0.3), atol=1e-15)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-15)
    assert best_id == TOKEN_TO_ID["car"]


if __name__ == "__main__":
    main()
