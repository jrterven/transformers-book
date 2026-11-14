"""Check Chapter 24's similarity axes, retrieval sets, and visual bridge."""

import math
import unittest

import numpy as np

from examples.ch24_multimodal.model import (
    bidirectional_recall_at_k,
    contrastive_logits,
    image_feature_bridge,
    symmetric_contrastive_loss,
)
from examples.ch24_multimodal.run_examples import (
    bridge_example,
    contrastive_example,
    retrieval_example,
)


class ContrastiveTests(unittest.TestCase):
    def test_worked_logits_and_two_direction_losses(self):
        report = contrastive_example()
        np.testing.assert_allclose(report["logits"], [[2, 1.2], [0, 1.6]])
        image_loss = (math.log1p(math.exp(-0.8)) + math.log1p(math.exp(-1.6))) / 2
        text_loss = (math.log1p(math.exp(-2)) + math.log1p(math.exp(-0.4))) / 2
        self.assertAlmostEqual(report["image_to_text_loss"], image_loss)
        self.assertAlmostEqual(report["text_to_image_loss"], text_loss)
        self.assertAlmostEqual(report["mean_loss"], (image_loss + text_loss) / 2)

    def test_feature_normalization_is_stable_across_scales(self):
        images = [[1e300, 0], [0, 1e-300], [1, 1]]
        texts = [[2e-300, 0], [3e299, 4e299]]
        expected = [[2, 1.2], [0, 1.6], [math.sqrt(2), 2.8 / math.sqrt(2)]]
        with np.errstate(all="raise"):
            logits = contrastive_logits(images, texts, 2)
        np.testing.assert_allclose(logits, expected, rtol=0, atol=1e-15)

    def test_diagonal_loss_is_stable_for_large_logits(self):
        self.assertEqual(symmetric_contrastive_loss([[0, 10_000], [10_000, 0]]),
                         (10_000, 10_000, 10_000))
        np.testing.assert_allclose(symmetric_contrastive_loss(np.full((3, 3), 1e300)),
                                   [math.log(3)] * 3)
        with self.assertRaisesRegex(ValueError, "square"):
            symmetric_contrastive_loss([[1, 2, 3], [4, 5, 6]])

    def test_checks_feature_width_zero_norm_and_scale(self):
        with self.assertRaisesRegex(ValueError, "share a width"):
            contrastive_logits([[1]], [[1, 0]], 2)
        with self.assertRaisesRegex(ValueError, "nonzero norm"):
            contrastive_logits([[0, 0]], [[1, 0]], 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            contrastive_logits([[1, 0]], [[1, 0]], 0)


class RetrievalTests(unittest.TestCase):
    def test_multiple_positives_and_missing_relevance_in_both_directions(self):
        report = retrieval_example()
        first, second = report["recall_at_1"], report["recall_at_2"]
        self.assertEqual(first["image_to_text"]["per_query"], [0.5, 1, None])
        self.assertEqual(first["text_to_image"]["per_query"], [1, 0.5, None])
        for name in ("image_to_text", "text_to_image"):
            self.assertEqual(first[name]["defined_queries"], 2)
            self.assertEqual(first[name]["mean"], 0.75)
            self.assertEqual(second[name]["mean"], 1)

    def test_ties_preserve_order_and_integer_scores_are_exact(self):
        relevant = [[0, 1]]
        tied = bidirectional_recall_at_k([[1, 1]], relevant, 1)
        self.assertEqual(tied["image_to_text"]["mean"], 0)
        large = np.array([[2**53, 2**53 + 1]], dtype=np.int64)
        distinct = bidirectional_recall_at_k(large, relevant, 1)
        self.assertEqual(distinct["image_to_text"]["mean"], 1)

    def test_empty_and_bad_relevance(self):
        report = bidirectional_recall_at_k(
            np.empty((2, 0)), np.empty((2, 0), dtype=bool), 2
        )
        self.assertEqual(report["image_to_text"]["per_query"], [None, None])
        self.assertEqual(report["text_to_image"]["per_query"], [])
        for direction in report.values():
            self.assertIsNone(direction["mean"])
            self.assertEqual(direction["defined_queries"], 0)
        with self.assertRaisesRegex(ValueError, "binary matrix"):
            bidirectional_recall_at_k([[0, 1]], [[1, 2]], 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            bidirectional_recall_at_k([[0, 1]], [[1, 0]], 0)


class BridgeTests(unittest.TestCase):
    def test_worked_image_attention_and_projection(self):
        result = bridge_example()
        self.assertEqual(result["token_shape"], [1, 3])
        np.testing.assert_allclose(result["attention_weights"], [[0.25, 0.75]])
        np.testing.assert_allclose(result["context"], [[0.5, 3]])
        np.testing.assert_allclose(result["language_width_tokens"], [[0.5, 3, -2.5]])

    def test_distinct_query_value_and_model_widths(self):
        q = [[0, 0], [0, 0]]
        k = [[0, 0], [1, 0], [0, 1]]
        v = [[1, 2, 3], [0, 1, 2], [2, 0, 1]]
        context, weights, tokens = image_feature_bridge(q, k, v, [[1], [2], [-1]])
        np.testing.assert_allclose(weights, np.full((2, 3), 1 / 3))
        np.testing.assert_allclose(context, [[1, 1, 2], [1, 1, 2]])
        np.testing.assert_allclose(tokens, [[1], [1]])
        with self.assertRaisesRegex(ValueError, "value width"):
            image_feature_bridge(q, k, v, [[1], [2]])


if __name__ == "__main__":
    unittest.main()
