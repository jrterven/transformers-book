"""Independent properties of the pretrained-model observations, without downloads."""
import unittest

import numpy as np

from examples.appendix_models.inspect_models import (
    cache_payload_bytes, stable_probabilities,
)


class AppendixModelFigureTests(unittest.TestCase):
    def test_normalization_preserves_a_known_probability_row_after_a_shift(self):
        logits = np.log([2., 1., 3.])
        expected = np.array([2., 1., 3.]) / 6
        np.testing.assert_allclose(stable_probabilities(logits + 1000), expected,
                                   rtol=0, atol=3e-14)

    def test_cross_entropy_derivative_matches_independent_finite_differences(self):
        logits = np.log([2., 1., 3.])
        p = stable_probabilities(logits)
        expected = p - np.array([0., 0., 1.])
        numerical = []
        for i in range(3):
            step = np.eye(3)[i] * 1e-5
            plus = -np.log(stable_probabilities(logits + step)[2])
            minus = -np.log(stable_probabilities(logits - step)[2])
            numerical.append((plus - minus) / 2e-5)
        np.testing.assert_allclose(numerical, expected, rtol=0, atol=2e-11)
        self.assertAlmostEqual(expected.sum(), 0)

    def test_payload_counts_keys_values_and_all_layers_once(self):
        # An asymmetric key/value width detects an accidental factor-of-two rule.
        self.assertEqual(cache_payload_bytes(3, 2, 4, 5, 2, 3, 2), 1200)
        self.assertEqual(cache_payload_bytes(6, 1, 12, 9, 64, 64, 4), 331776)
        self.assertEqual(cache_payload_bytes(6, 1, 12, 18, 64, 64, 4), 663552)

    def test_invalid_rows_and_nonpositive_dimensions_are_rejected(self):
        for row in ([], [np.inf], [np.nan], [[1, 2]]):
            with self.assertRaises(ValueError):
                stable_probabilities(row)
        with self.assertRaises(ValueError):
            cache_payload_bytes(6, 1, 12, 0, 64, 64, 4)


if __name__ == '__main__':
    unittest.main()
