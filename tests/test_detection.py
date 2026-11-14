"""Check Chapter 23's IoU conventions, set matches, and AP."""

import unittest

import numpy as np

from examples.ch23_detection.model import (
    all_points_ap,
    box_iou,
    mask_iou,
    match_queries,
    precision_recall,
)
from examples.ch23_detection.run_examples import ap_example, assignment_example, overlap_example


class OverlapTests(unittest.TestCase):
    def test_worked_iou_and_continuous_corners(self):
        result = overlap_example()
        self.assertEqual(result, {"box_iou": 1 / 3, "mask_iou": 0.5})
        self.assertEqual(box_iou([0, 0, 2, 2], [2, 0, 3, 2]), 0)
        self.assertEqual(box_iou([0, 0, 2, 2], [0.25, 0.5, 1.25, 1.5]), 0.25)
        self.assertEqual(box_iou([0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]), 1)

    def test_mask_empty_union_and_grid_validation(self):
        empty = [[False, False], [False, False]]
        self.assertIsNone(mask_iou(empty, empty))
        self.assertEqual(mask_iou(empty, [[True, False], [False, False]]), 0)
        with self.assertRaisesRegex(ValueError, "Boolean"):
            mask_iou([[0, 1]], [[0, 1]])
        with self.assertRaisesRegex(ValueError, "same"):
            mask_iou([[True, False]], [[True]])

    def test_boxes_require_finite_positive_corners(self):
        valid = [0, 0, 2, 2]
        with self.assertRaisesRegex(ValueError, "positive"):
            box_iou([0, 0, 0, 2], valid)
        with self.assertRaisesRegex(ValueError, "finite real"):
            box_iou([0, 0, np.inf, 1], valid)
        with self.assertRaisesRegex(ValueError, "shape"):
            box_iou([0, 0, 2], [0, 0, 2])


class AssignmentTests(unittest.TestCase):
    def test_worked_assignment_resolves_conflicting_minima(self):
        self.assertEqual(match_queries([[1, 1], [2, 5], [4, 3]]),
                         (((1, 0), (0, 1)), (2,), 3))
        report = assignment_example()
        self.assertEqual(report["pairs"], [["Q2", "A"], ["Q1", "B"]])
        self.assertEqual(report["unmatched"], ["Q3"])

    def test_assignment_ties_empty_targets_and_scope(self):
        self.assertEqual(match_queries(np.zeros((3, 2))),
                         (((0, 0), (1, 1)), (2,), 0))
        self.assertEqual(match_queries(np.empty((3, 0))), ((), (0, 1, 2), 0))
        with self.assertRaisesRegex(ValueError, "at least as many"):
            match_queries(np.zeros((1, 2)))
        with self.assertRaisesRegex(ValueError, "100,000"):
            match_queries(np.zeros((9, 9)))

    def test_assignment_preserves_large_integer_cost_order(self):
        large = 2**53
        cost = np.array([[large + 1], [large]], dtype=np.int64)
        self.assertEqual(match_queries(cost), (((1, 0),), (0,), large))


class RankedDetectionTests(unittest.TestCase):
    def test_duplicate_is_false_positive_and_ap_uses_all_recall_increases(self):
        report = ap_example()
        self.assertAlmostEqual(report["duplicate_iou_with_a"], 0.6)
        self.assertEqual(report["true_positive"], [True, False, True])
        np.testing.assert_allclose(report["precision"], [1, 0.5, 2 / 3])
        np.testing.assert_array_equal(report["recall"], [0.5, 0.5, 1])
        self.assertAlmostEqual(report["all_points_ap"], 5 / 6)

    def test_rank_ties_and_matching_the_best_unused_target(self):
        a, b = [0, 0, 2, 2], [1, 0, 3, 2]
        order, flags, _, _ = precision_recall([a, b], [a, a], [0.7, 0.7], threshold=1 / 3)
        np.testing.assert_array_equal(order, [0, 1])
        np.testing.assert_array_equal(flags, [True, True])

    def test_empty_target_and_empty_prediction_policies(self):
        box = [0, 0, 2, 2]
        _, flags, precision, recall = precision_recall([], [box], [0.7])
        np.testing.assert_array_equal(flags, [False])
        np.testing.assert_array_equal(precision, [0])
        self.assertIsNone(recall)
        self.assertIsNone(all_points_ap(precision, recall))
        _, _, precision, recall = precision_recall([box], [], [])
        self.assertEqual(precision.size, 0)
        self.assertEqual(recall.size, 0)
        self.assertEqual(all_points_ap(precision, recall), 0)

    def test_ap_envelope_preserves_incomplete_recall_and_validates_curve(self):
        self.assertAlmostEqual(all_points_ap([0, 0.5, 2 / 3], [0, 0.5, 1]), 2 / 3)
        self.assertEqual(all_points_ap([1, 0.5], [0.25, 0.25]), 0.25)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            all_points_ap([0.5, 0.5], [0.9, 0.8])
        with self.assertRaisesRegex(ValueError, "one score"):
            precision_recall([], [[0, 0, 2, 2]], [])


if __name__ == "__main__":
    unittest.main()
