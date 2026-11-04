"""Numerical and information-flow checks for the Chapter 1 teaching model."""

import math
import unittest

import numpy as np

from examples.ch01_language.model import (
    BOS, EOS, PAD, TOKEN_TO_ID, VOCAB, context_logits, embedding_lookup,
    next_token_pairs, stable_softmax, token_nll, toy_parameters,
)


class LanguageModelTests(unittest.TestCase):
    def setUp(self):
        self.parameters = toy_parameters()
        self.ids = np.array((4, 12, 7))

    def test_lookup_preserves_sequence_order_and_repeated_rows(self):
        got = embedding_lookup(self.parameters["E"], [12, 4, 12, 7])
        expected = np.array([[-0.3, 0.8, 0.2, -0.1], [0.2, -0.4, 0.1, 0.5],
                             [-0.3, 0.8, 0.2, -0.1], [0.6, 0.1, -0.5, 0.3]])
        np.testing.assert_array_equal(got, expected)
        self.assertEqual(got.shape, (4, 4))

    def test_renumbering_ids_and_table_together_preserves_vectors(self):
        permutation = np.roll(np.arange(16), 5)
        inverse = np.argsort(permutation)
        renamed_table = self.parameters["E"][permutation]
        original = embedding_lookup(self.parameters["E"], self.ids)
        renamed = embedding_lookup(renamed_table, inverse[self.ids])
        np.testing.assert_array_equal(renamed, original)

    def test_worked_context_and_full_vocabulary_probabilities(self):
        hidden, logits = context_logits([4, 12], self.parameters)
        np.testing.assert_allclose(hidden, [0, 1, 0.5, 0.3], atol=1e-15)
        expected_logits = np.zeros(16)
        expected_logits[7], expected_logits[12] = 2, 1
        np.testing.assert_allclose(logits, expected_logits, atol=1e-15)
        denominator = math.exp(2) + math.exp(1) + 14
        probabilities = stable_softmax(logits)
        self.assertEqual(probabilities.shape, (16,))
        self.assertAlmostEqual(probabilities[7], math.exp(2) / denominator, places=15)
        self.assertAlmostEqual(probabilities[12], math.exp(1) / denominator, places=15)
        self.assertAlmostEqual(probabilities[4], 1 / denominator, places=15)
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=15)
        self.assertEqual(int(probabilities.argmax()), TOKEN_TO_ID["car"])

    def test_context_model_uses_token_order_but_lookup_is_static(self):
        forward, forward_logits = context_logits([4, 12], self.parameters)
        backward, backward_logits = context_logits([12, 4], self.parameters)
        np.testing.assert_allclose(backward, [0.1, 0.0, 0.4, 0.9], atol=1e-15)
        np.testing.assert_array_equal(backward_logits, np.zeros(16))
        np.testing.assert_allclose(stable_softmax(backward_logits), np.full(16, 1 / 16))
        self.assertFalse(np.allclose(forward_logits, backward_logits))
        np.testing.assert_array_equal(embedding_lookup(self.parameters["E"], [12])[0],
                                      [-0.3, 0.8, 0.2, -0.1])

    def test_softmax_is_stable_and_normalizes_vocabulary_not_batch(self):
        logits = np.array([[1002., 1001., 1000.], [-1000., -1001., -1002.]])
        probabilities = stable_softmax(logits)
        expected_row = np.exp([2., 1., 0.]) / np.exp([2., 1., 0.]).sum()
        np.testing.assert_allclose(probabilities, np.stack([expected_row] * 2), atol=1e-15)
        np.testing.assert_allclose(probabilities.sum(axis=-1), [1, 1], atol=1e-15)
        np.testing.assert_allclose(stable_softmax(logits + 1000), probabilities)
        with self.assertRaises(ValueError):
            stable_softmax([0.0, float("nan")])

    def test_training_pairs_predict_following_token_and_final_eos(self):
        contexts, targets = next_token_pairs(self.ids)
        np.testing.assert_array_equal(contexts, [[BOS, BOS], [BOS, 4], [4, 12], [12, 7]])
        np.testing.assert_array_equal(targets, [4, 12, 7, EOS])
        self.assertEqual(contexts.shape, (4, 2))
        self.assertFalse(np.isin(targets, [BOS, PAD]).any())
        with self.assertRaises(ValueError):
            next_token_pairs([4, PAD, 7])
        with self.assertRaises(ValueError):
            next_token_pairs([])

    def test_negative_log_probability_matches_manual_loss(self):
        _, logits = context_logits([4, 12], self.parameters)
        expected = math.log(math.exp(2) + math.e + 14) - 2
        loss = token_nll(logits, np.array(7))
        self.assertAlmostEqual(float(loss), expected, places=15)
        raised = logits.copy()
        raised[7] += 1
        self.assertLess(float(token_nll(raised, np.array(7))), float(loss))
        # A finite but extremely unlikely target must not underflow via log(softmax).
        self.assertAlmostEqual(float(token_nll([1000., -1000.], np.array(1))), 2000.0)

    def test_batched_shapes_and_id_validation(self):
        batch_ids = np.stack((self.ids, self.ids[::-1]))
        self.assertEqual(embedding_lookup(self.parameters["E"], batch_ids).shape, (2, 3, 4))
        hidden, logits = context_logits(batch_ids[:, :2], self.parameters)
        self.assertEqual(hidden.shape, (2, 4))
        self.assertEqual(logits.shape, (2, 16))
        for row in range(2):
            h, z = context_logits(batch_ids[row, :2], self.parameters)
            np.testing.assert_allclose(hidden[row], h)
            np.testing.assert_allclose(logits[row], z)
        for ids, error in [([-1], ValueError), ([16], ValueError), ([4.0], TypeError)]:
            with self.assertRaises(error):
                embedding_lookup(self.parameters["E"], ids)
        with self.assertRaises(ValueError):
            context_logits([4, 12, 7], self.parameters)


if __name__ == "__main__":
    unittest.main()
