"""Check Chapter 27 windows, seasonal fits, and horizon metrics."""

from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from examples.ch27_time_series.demand_data import AREAS, SNAPSHOT, load_snapshot
from examples.ch27_time_series.model import (
    fit_seasonal_drift,
    forecast_metrics,
    rolling_windows,
    seasonal_drift,
    seasonal_naive,
)
from examples.ch27_time_series.run_examples import interval_example, rolling_example, series_example
from examples.ch27_time_series.run_pretrained import build_replay, score_quantiles


class WindowTests(unittest.TestCase):
    def test_origin_is_first_unknown_and_only_complete_windows_are_returned(self):
        origins, past, future = rolling_windows(np.arange(8), 3, 2)
        np.testing.assert_array_equal(origins, [3, 4, 5, 6])
        np.testing.assert_array_equal(past, [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
        np.testing.assert_array_equal(future, [[3, 4], [4, 5], [5, 6], [6, 7]])

    def test_empty_window_shapes_and_validation(self):
        origins, past, future = rolling_windows([1, 2], 3, 4)
        self.assertEqual(origins.shape, (0,))
        self.assertEqual(past.shape, (0, 3))
        self.assertEqual(future.shape, (0, 4))
        with self.assertRaisesRegex(ValueError, "positive"):
            rolling_windows([1, 2], 2, 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            rolling_windows([1, np.nan], 1, 1)

    def test_train_and_test_targets_are_disjoint_at_the_boundary(self):
        report = rolling_example()
        self.assertEqual(report["training_origins"], list(range(4, 22)))
        self.assertEqual(report["omitted_crossing_origins"], [22, 23])
        self.assertEqual(report["test_origins"], list(range(24, 30)))
        horizon, split = report["horizon"], report["split_index"]
        train_targets = {origin + h for origin in report["training_origins"] for h in range(horizon)}
        test_targets = {origin + h for origin in report["test_origins"] for h in range(horizon)}
        self.assertLess(max(train_targets), split)
        self.assertEqual(min(test_targets), split)
        self.assertFalse(train_targets & test_targets)

    def test_future_perturbations_change_targets_but_leave_origin_inputs_alone(self):
        values = series_example()
        changed = values.copy()
        changed[25:] += 1000
        origins, past, future = rolling_windows(values, 4, 3)
        _, changed_past, changed_future = rolling_windows(changed, 4, 3)
        row = int(np.flatnonzero(origins == 25)[0])
        np.testing.assert_array_equal(past[row], changed_past[row])
        self.assertEqual(past[row, -1], values[24])
        np.testing.assert_array_equal(changed_future[row], future[row] + 1000)
        fitted = fit_seasonal_drift(values[:24], 4)
        self.assertEqual(fit_seasonal_drift(changed[:24], 4), fitted)
        np.testing.assert_array_equal(
            seasonal_drift(values[:25], 4, 3, fitted),
            seasonal_drift(changed[:25], 4, 3, fitted),
        )


class SeasonalTests(unittest.TestCase):
    def test_worked_rolling_metrics(self):
        report = rolling_example()
        self.assertEqual(report["targets"][0], [24, 28, 25])
        self.assertEqual(report["seasonal_naive"][0], [20, 24, 21])
        self.assertEqual(report["fitted_drift"], 4)
        self.assertEqual(report["seasonal_drift"], report["targets"])
        self.assertEqual(report["mae_seasonal_naive"], [4, 4, 4])
        self.assertEqual(report["mae_seasonal_drift"], [0, 0, 0])

    def test_drift_fit_and_forecast_over_several_cycles(self):
        learned = fit_seasonal_drift([1, 4, 2, 7, 4, 10], 2)
        self.assertEqual(learned, 2.25)
        history = [1, 4, 2, 7]
        np.testing.assert_array_equal(seasonal_naive(history, 2, 5), [2, 7, 2, 7, 2])
        np.testing.assert_array_equal(
            seasonal_drift(history, 2, 5, learned), [4.25, 9.25, 6.5, 11.5, 8.75]
        )
        np.testing.assert_array_equal(seasonal_drift([3, 2], 1, 3, -1), [1, 0, -1])

    def test_baselines_validate_lengths_and_fitted_coefficient(self):
        with self.assertRaisesRegex(ValueError, "observed period"):
            seasonal_naive([1, 2], 3, 1)
        with self.assertRaisesRegex(ValueError, "lagged pair"):
            fit_seasonal_drift([1, 2], 2)
        with self.assertRaisesRegex(ValueError, "drift must"):
            seasonal_drift([1, 2], 2, 2, [1])


class MetricTests(unittest.TestCase):
    def test_worked_horizon_metrics(self):
        report = interval_example()
        self.assertEqual(report["mae"], [0.5, 0.5, 1.5])
        self.assertEqual(report["coverage"], [1, 1, 0.5])
        self.assertEqual(report["mean_interval_width"], [2, 2, 2])

    def test_inclusive_coverage_and_zero_width(self):
        mae, coverage, width = forecast_metrics(
            [[3, 5], [2, 6]], [[3, 4], [2, 6]], [[3, 4], [2, 4]], [[3, 5], [3, 6]]
        )
        np.testing.assert_array_equal(mae, [0, 0.5])
        np.testing.assert_array_equal(coverage, [1, 1])
        np.testing.assert_array_equal(width, [0.5, 1.5])

    def test_metric_inputs_must_align_and_intervals_must_be_ordered(self):
        with self.assertRaisesRegex(ValueError, "same nonempty"):
            forecast_metrics([[1], [2]], [[1]], [[0]], [[2]])
        with self.assertRaisesRegex(ValueError, "lower endpoints"):
            forecast_metrics([[1]], [[1]], [[2]], [[1]])
        with self.assertRaisesRegex(ValueError, "same nonempty"):
            forecast_metrics(*(np.empty((0, 3)) for _ in range(4)))


class DemandReplayTests(unittest.TestCase):
    def test_issue_cutoff_and_calibration_targets_respect_reporting_lag(self):
        hours, demand = load_snapshot(SNAPSHOT)
        replay = build_replay(hours, demand["NYIS"], AREAS["NYIS"])
        first_issue = datetime(2026, 10, 30, tzinfo=timezone.utc)
        self.assertEqual(replay["test_issues"][0], first_issue)
        self.assertEqual(replay["history"].shape, (14, 672))
        self.assertEqual(replay["actual"].shape, (14, 24))
        idx = hours.index(first_issue)
        changed = demand["NYIS"].copy()
        changed[idx:] += 100_000
        perturbed = build_replay(hours, changed, AREAS["NYIS"])
        np.testing.assert_array_equal(replay["history"][0], demand["NYIS"][idx - 672:idx])
        np.testing.assert_array_equal(perturbed["history"][0], replay["history"][0])
        np.testing.assert_array_equal(replay["actual"][0], demand["NYIS"][idx + 1:idx + 25])
        np.testing.assert_array_equal(perturbed["actual"][0], replay["actual"][0] + 100_000)
        last_arrival = replay["calibration_issues"][-1] + timedelta(hours=25)
        self.assertLess(last_arrival, first_issue)

    def test_calendar_aligns_to_hour_start_across_fall_clock_change(self):
        hours, demand = load_snapshot(SNAPSHOT)
        replay = build_replay(hours, demand["NYIS"], AREAS["NYIS"])
        change_day = datetime(2026, 11, 1, tzinfo=timezone.utc)
        calendar = replay["calendar"][replay["test_issues"].index(change_day)]
        self.assertEqual(calendar.shape, (3, 697))
        for ending_hour in (6, 7):
            np.testing.assert_allclose(calendar[:, 672 + ending_hour],
                                       [np.sin(np.pi / 12), np.cos(np.pi / 12), 1])
        np.testing.assert_allclose(calendar[:, 672 + 8], [0.5, np.sqrt(3) / 2, 1])

    def test_scoring_skips_gap_and_charges_for_interval_miss(self):
        output = np.array([[[0, 1, 2], [11, 12, 13], [8, 4, 6]]])
        scored = score_quantiles(output, (0.1, 0.5, 0.9), np.array([[14, 4]]))
        self.assertEqual(scored["raw_crossing_slots"], 1)
        self.assertEqual(scored["median_mw"], [[12, 6]])
        self.assertEqual(scored["metrics"]["interval_score_by_hour_mw"], [12, 4])
        self.assertEqual(scored["metrics"]["coverage_by_hour"], [0, 1])


if __name__ == "__main__":
    unittest.main()
