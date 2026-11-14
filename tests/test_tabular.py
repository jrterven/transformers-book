"""Check Chapter 29's train-only encoder and probability calculations."""

import math
import unittest

import numpy as np

from examples.ch29_tabular.model import (
    brier_score,
    context_probability,
    encode_rows,
    fit_row_encoder,
    histogram_summary,
)
from examples.ch29_tabular.run_examples import (
    TRAIN_CATEGORIES,
    TRAIN_NUMERIC,
    probability_example,
    row_example,
)
from examples.ch29_tabular.tabular_data import load_tables, recipe_groups


class EncodingTests(unittest.TestCase):
    def test_worked_rows_use_sorted_training_levels(self):
        result = row_example()
        self.assertEqual(result["median"], 4)
        self.assertEqual(result["levels"], ("shop", "web"))
        self.assertEqual(result["columns"], (
            "numeric_filled", "numeric_missing", "shop", "web",
            "category_missing", "category_unknown",
        ))
        self.assertEqual(result["train_features"], [
            [2, 0, 0, 1, 0, 0], [4, 0, 1, 0, 0, 0],
            [4, 1, 0, 0, 1, 0], [8, 0, 0, 1, 0, 0],
        ])
        self.assertEqual(result["held_out_features"], [
            [4, 1, 0, 0, 0, 1], [6, 0, 1, 0, 0, 0], [100, 0, 0, 0, 1, 0],
        ])

    def test_held_out_values_do_not_refit_the_schema(self):
        fit = fit_row_encoder(TRAIN_NUMERIC, TRAIN_CATEGORIES)
        original = encode_rows([None], ["kiosk"], *fit)
        enlarged = encode_rows([None, -1e6, 1e6], ["kiosk", "new", "shop"], *fit)
        np.testing.assert_array_equal(enlarged[:1], original)
        self.assertEqual(fit, (4, ("shop", "web")))

    def test_numeric_none_and_nan_and_category_none_are_distinct(self):
        fit = fit_row_encoder([np.nan, 2, 4], ["None", None, "None"])
        self.assertEqual(fit, (3, ("None",)))
        features = encode_rows([None, np.nan, 0], ["None", None, "other"], *fit)
        np.testing.assert_array_equal(features, [
            [3, 1, 1, 0, 0], [3, 1, 0, 1, 0], [0, 0, 0, 0, 1],
        ])

    def test_empty_transform_and_unobserved_training_categories(self):
        fit = fit_row_encoder([2, 4], [None, None])
        self.assertEqual(fit, (3, ()))
        self.assertEqual(encode_rows([], [], *fit).shape, (0, 4))
        np.testing.assert_array_equal(
            encode_rows([None, 2], ["web", None], *fit), [[3, 1, 0, 1], [2, 0, 1, 0]]
        )

    def test_schema_rejects_bad_columns_levels_and_fit_data(self):
        with self.assertRaisesRegex(ValueError, "row count"):
            fit_row_encoder([1, 2], ["web"])
        with self.assertRaisesRegex(ValueError, "observed numeric"):
            fit_row_encoder([None, np.nan], ["web", None])
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_row_encoder([np.inf], ["web"])
        with self.assertRaisesRegex(ValueError, "real"):
            fit_row_encoder(["1"], ["web"])
        with self.assertRaisesRegex(ValueError, "unique"):
            encode_rows([1], ["web"], 1, ["web", "web"])
        with self.assertRaisesRegex(ValueError, "median"):
            encode_rows([1], ["web"], np.nan, [])


class ProbabilityTests(unittest.TestCase):
    def test_worked_context_probability_and_brier(self):
        result = probability_example()
        np.testing.assert_allclose(result["context_weights"], [1 / 6, 1 / 3, 1 / 2])
        self.assertAlmostEqual(result["context_probability"], 5 / 6)
        self.assertAlmostEqual(result["brier_score"], 0.07)

    def test_context_score_uses_stable_normalization(self):
        with np.errstate(all="raise"):
            p, weights = context_probability([-1e308, 1e308], [1, 0])
        self.assertEqual(p, 0)
        np.testing.assert_array_equal(weights, [0, 1])
        p, _ = context_probability(np.array([0, math.log(2), math.log(3)]) + 1e4,
                                   [0, 1, 1])
        self.assertAlmostEqual(p, 5 / 6, places=12)
        p, _ = context_probability([0.2, 0.8, -1], [1, 1, 1])
        self.assertAlmostEqual(p, 1)

    def test_brier_and_context_validate_labels_and_aligned_inputs(self):
        self.assertEqual(brier_score([False, True], [0, 1]), 0)
        self.assertEqual(brier_score([0, 1], [1, 0]), 1)
        with self.assertRaisesRegex(ValueError, "binary"):
            brier_score([0, 2], [0.2, 0.4])
        with self.assertRaisesRegex(ValueError, "one real probability"):
            brier_score([0, 1], [[0.2], [0.4]])
        with self.assertRaisesRegex(ValueError, "finite and between"):
            brier_score([0, 1], [0.2, 1.4])
        with self.assertRaisesRegex(ValueError, "one real similarity"):
            context_probability([0, 1], [1])
        with self.assertRaisesRegex(ValueError, "one real similarity"):
            context_probability([1j, 0], [0, 1])


class PretrainedDemoMathTests(unittest.TestCase):
    def test_recipe_groups_ignore_age_but_keep_ingredient_changes(self):
        rows = np.array([
            [525, 0, 0, 189, 0, 1125, 613, 3],
            [525, 0, 0, 189, 0, 1125, 613, 28],
            [525, 0, 0, 190, 0, 1125, 613, 3],
        ])
        groups = recipe_groups(rows)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        tables = load_tables()
        self.assertEqual(tables["raisin"].x.shape, (900, 7))
        self.assertEqual(tables["concrete"].x.shape, (1005, 8))
        self.assertEqual(len(set(tables["concrete"].groups)), 427)

    def test_histogram_quantiles_interpolate_and_can_skip_empty_bins(self):
        mean, quantiles = histogram_summary(
            [10, 30, 50, 70], [[0.25, 0.5, 0.25], [0.25, 0, 0.75]],
            [0.1, 0.25, 0.5, 0.9],
        )
        np.testing.assert_allclose(mean, [40, 50])
        np.testing.assert_allclose(
            quantiles, [[18, 30, 40, 62], [18, 30, 50 + 20/3, 50 + 52/3]]
        )


if __name__ == "__main__":
    unittest.main()
