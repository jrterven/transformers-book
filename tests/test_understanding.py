"""Check Chapter 19 pooling, first-piece supervision, and BIO spans."""

import unittest

import numpy as np

from examples.ch19_understanding.model import (
    bio_spans,
    first_piece_labels,
    first_piece_predictions,
    sentence_logits,
    span_metrics,
)
from examples.ch19_understanding.run_examples import entity_example, pooling_example


class PoolingTests(unittest.TestCase):
    def test_worked_pool_and_head(self):
        self.assertEqual(pooling_example(), {"pooled": [1, 1], "logits": [0, 1.25]})

    def test_batch_masks_each_sentence_and_excludes_invalid_padding(self):
        features = np.array([
            [[1, 0], [np.nan, np.inf], [2, 1]],
            [[np.nan, np.nan], [3, 4], [np.inf, 1]],
        ])
        valid = np.array([[True, False, True], [False, True, False]])
        weight = [[2, -1], [0, 1]]
        bias = [1, -1]
        with np.errstate(all="raise"):
            pooled, logits = sentence_logits(features, valid, weight, bias)
        np.testing.assert_array_equal(pooled, [[1.5, 0.5], [3, 4]])
        np.testing.assert_array_equal(logits, [[3.5, -0.5], [3, 3]])

    def test_pooling_rejects_bad_shapes_masks_and_selected_values(self):
        x, w, b = [[1, 0], [2, 3]], [[0, 1]], [0]
        with self.assertRaisesRegex(ValueError, "boolean"):
            sentence_logits(x, [1, 0], w, b)
        with self.assertRaisesRegex(ValueError, "at least one"):
            sentence_logits(x, [False, False], w, b)
        with self.assertRaisesRegex(ValueError, "classes"):
            sentence_logits(x, [True, True], [[1, 2, 3]], b)
        with self.assertRaisesRegex(ValueError, "finite"):
            sentence_logits([[1, np.nan], [2, 3]], [True, True], w, b)


class AlignmentTests(unittest.TestCase):
    def test_lesson_alignment_and_prediction_recovery(self):
        result = entity_example()
        self.assertEqual(result["token_labels"], [-100, 1, 0, 0, 2, 3, -100, 0, -100, -100])
        self.assertEqual(result["predicted_tags"], ["B-PER", "O", "O", "B-LOC", "O", "O"])

    def test_first_piece_rules_with_custom_ignore_and_empty_words(self):
        ids = [-1, 0, 0, 1, -1, 2, 2]
        np.testing.assert_array_equal(
            first_piece_labels(ids, [3, 4, 5], ignore=-9), [-9, 3, -9, 4, -9, 5, -9]
        )
        np.testing.assert_array_equal(
            first_piece_predictions(ids, [-99, 3, -99, 4, -99, 5, -99]), [3, 4, 5]
        )
        np.testing.assert_array_equal(first_piece_labels([-1, -1], []), [-100, -100])
        self.assertEqual(first_piece_predictions([-1, -1], [0, 0]).shape, (0,))
        self.assertEqual(first_piece_labels([], []).shape, (0,))

    def test_alignment_rejects_discontinuous_words_and_mismatched_labels(self):
        for ids in ([0, 2], [0, 1, 0], [0, -1, 0]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "consecutive"):
                first_piece_labels(ids, [0, 0])
        with self.assertRaisesRegex(ValueError, "special"):
            first_piece_labels([-2, 0], [0])
        with self.assertRaisesRegex(ValueError, "one nonnegative tag"):
            first_piece_labels([0, 1], [0])
        with self.assertRaisesRegex(ValueError, "each token"):
            first_piece_predictions([0, 1], [0])


class SpanTests(unittest.TestCase):
    def test_lesson_requires_exact_boundary_and_type(self):
        report = entity_example()
        self.assertEqual(report["gold_spans"], [(0, 1, "PER"), (3, 5, "LOC")])
        self.assertEqual(report["predicted_spans"], [(0, 1, "PER"), (3, 4, "LOC")])
        self.assertEqual(report["metrics"], {
            "correct_spans": 1, "gold_spans": 2, "predicted_spans": 2,
            "precision": 0.5, "recall": 0.5, "f1": 0.5, "word_accuracy": 5 / 6,
        })
        self.assertEqual(span_metrics(["B-LOC"], ["B-PER"])["f1"], 0)

    def test_strict_decoding_and_optional_inside_tag_repair(self):
        tags = ["I-PER", "I-PER", "I-LOC", "O"]
        with self.assertRaisesRegex(ValueError, "unexpected inside tag"):
            bio_spans(tags)
        with self.assertRaisesRegex(ValueError, "unexpected inside tag"):
            bio_spans(["B-PER", "I-LOC"])
        self.assertEqual(bio_spans(tags, repair=True), {(0, 2, "PER"), (2, 3, "LOC")})
        self.assertEqual(bio_spans(["B-LOC", "B-LOC"]), {(0, 1, "LOC"), (1, 2, "LOC")})
        repaired = span_metrics(["B-LOC"], ["I-LOC"], repair_predictions=True)
        self.assertEqual(repaired["f1"], 1)
        self.assertEqual(repaired["word_accuracy"], 0)
        with self.assertRaisesRegex(ValueError, "invalid BIO"):
            bio_spans(["X-LOC"], repair=True)
        with self.assertRaisesRegex(ValueError, "unexpected inside tag"):
            span_metrics(["I-LOC"], ["B-LOC"], repair_predictions=True)

    def test_empty_policy_and_alignment(self):
        no_spans = span_metrics(["O"], ["O"])
        self.assertEqual(no_spans["word_accuracy"], 1)
        for name in ("precision", "recall", "f1"):
            self.assertIsNone(no_spans[name])
        one_side = span_metrics(["B-LOC"], ["O"])
        self.assertIsNone(one_side["precision"])
        self.assertEqual(one_side["recall"], 0)
        self.assertEqual(one_side["f1"], 0)
        self.assertIsNone(span_metrics([], [])["word_accuracy"])
        with self.assertRaisesRegex(ValueError, "same words"):
            span_metrics(["O"], [])


if __name__ == "__main__":
    unittest.main()
