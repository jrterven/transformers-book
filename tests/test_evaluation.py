"""Check the Chapter 17 splits, metrics, bins, and paired comparisons."""

import unittest

import numpy as np

from examples.ch17_evaluation.model import (
    binary_metrics,
    paired_accuracy_bootstrap,
    reliability_bins,
    split_by_group,
)
from examples.ch17_evaluation.run_examples import (
    bootstrap_example,
    calibration_example,
    classification_data,
    metric_example,
)


class SplitTests(unittest.TestCase):
    def test_split_reproducibly_keeps_every_group_together(self):
        row_ids, groups, *_ = classification_data()
        first = split_by_group(groups, (3, 1, 2))
        second = split_by_group(groups, (3, 1, 2))
        self.assertEqual(tuple(map(len, first)), (8, 1, 3))
        expected_groups = ({"a", "b", "e"}, {"c"}, {"d", "f"})
        for part, repeat, expected in zip(first, second, expected_groups):
            np.testing.assert_array_equal(part, repeat)
            self.assertEqual(set(groups[part]), expected)
        np.testing.assert_array_equal(np.sort(row_ids[np.concatenate(first)]), row_ids)

    def test_group_assignment_uses_ids_and_survives_row_reordering(self):
        groups = np.array([7, 2, 7, 5, 9, 9])
        original = split_by_group(groups, (1, 1, 2))
        reverse = split_by_group(groups[::-1], (1, 1, 2))
        for a, b in zip(original, reverse):
            self.assertEqual(set(groups[a]), set(groups[::-1][b]))

    def test_invalid_group_split_inputs(self):
        with self.assertRaisesRegex(ValueError, "counts"):
            split_by_group([1, 1, 2, 3], (1, 1, 2))
        with self.assertRaisesRegex(ValueError, "empty strings"):
            split_by_group(["a", "", "b"], (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "integers or strings"):
            split_by_group([1, np.nan, 3], (1, 1, 1))


class MetricTests(unittest.TestCase):
    def test_accuracy_tie_has_different_precision_and_recall(self):
        result = metric_example()
        keys = ("tp", "fn", "fp", "tn")
        a, b = result["model_a"], result["model_b"]
        self.assertEqual([a[key] for key in keys], [2, 2, 1, 7])
        self.assertEqual([b[key] for key in keys], [3, 1, 2, 6])
        self.assertEqual(a["accuracy"], b["accuracy"])
        self.assertEqual(a["accuracy"], 0.75)
        self.assertEqual((a["precision"], b["precision"]), (2 / 3, 3 / 5))
        self.assertEqual((a["recall"], b["recall"]), (1 / 2, 3 / 4))
        self.assertEqual((a["f1"], b["f1"]), (4 / 7, 2 / 3))

    def test_undefined_binary_metrics_and_bad_alignment(self):
        result = binary_metrics([0, 0], [0, 0])
        self.assertEqual(result["specificity"], 1)
        for metric in ("precision", "recall", "f1"):
            self.assertIsNone(result[metric])
        with self.assertRaisesRegex(ValueError, "same rows"):
            binary_metrics([0, 1], [0])
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            binary_metrics([0, 1], [0.3, 1])

    def test_calibration_fixture(self):
        result = calibration_example()
        self.assertEqual(result["counts"], [4, 4, 4])
        np.testing.assert_allclose(result["mean_scores"], [0.2, 0.5, 0.8])
        np.testing.assert_array_equal(result["positive_rates"], [0.25, 0.5, 0.75])
        self.assertAlmostEqual(result["brier_score"], 2.38 / 12)

    def test_calibration_edges_empty_bins_and_invalid_probabilities(self):
        counts, means, rates, brier = reliability_bins(
            [0, 1, 0, 1], [0, 0.25, 1, 1], [0, 0.25, 0.5, 0.75, 1]
        )
        np.testing.assert_array_equal(counts, [1, 1, 0, 2])
        np.testing.assert_array_equal(means, [0, 0.25, np.nan, 1])
        np.testing.assert_array_equal(rates, [0, 1, np.nan, 0.5])
        self.assertEqual(brier, 0.390625)
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            reliability_bins([0, 1], [0.5, 0.5], [0, 0.5, 0.5, 1])
        with self.assertRaisesRegex(ValueError, "finite"):
            reliability_bins([0, 1], [0.5, np.nan], [0, 1])


class BootstrapTests(unittest.TestCase):
    def test_identical_predictions_cancel_in_every_draw(self):
        truth, pred = [1, 0, 0, 1], [0, 0, 1, 1]
        for groups in (None, ["a", "b", "b", "b"]):
            self.assertEqual(
                paired_accuracy_bootstrap(truth, pred, pred, groups=groups),
                (0, (0, 0)),
            )

    def test_group_draw_matches_repeated_whole_rows_with_unequal_group_sizes(self):
        _, groups, y, a, b = classification_data()
        names = sorted(set(groups))
        draws = np.random.default_rng(83).integers(0, len(names), size=(64, len(names)))
        reference = []
        for replicate in draws:
            rows = [i for group in replicate for i, label in enumerate(groups)
                    if label == names[group]]
            reference.append(np.mean(b[rows] == y[rows]) - np.mean(a[rows] == y[rows]))
        point, interval = paired_accuracy_bootstrap(
            y, a, b, repeats=64, confidence=0.8, seed=83, groups=groups
        )
        self.assertEqual(point, 0)
        np.testing.assert_allclose(interval, np.quantile(reference, [0.1, 0.9]))

    def test_seeded_fixture_row_and_group_intervals(self):
        report = bootstrap_example()
        self.assertEqual(report["rows"]["point"], 0)
        self.assertEqual(report["groups"]["point"], 0)
        np.testing.assert_allclose(report["rows"]["interval"], [-5 / 12, 5 / 12])
        np.testing.assert_allclose(report["groups"]["interval"], [-5 / 9, 7 / 13])

    def test_bootstrap_validation(self):
        y, a, b = [1, 0], [0, 0], [1, 1]
        with self.assertRaisesRegex(ValueError, "positive"):
            paired_accuracy_bootstrap(y, a, b, repeats=0)
        with self.assertRaisesRegex(ValueError, "one resampling group"):
            paired_accuracy_bootstrap(y, a, b, groups=[1])


if __name__ == "__main__":
    unittest.main()
