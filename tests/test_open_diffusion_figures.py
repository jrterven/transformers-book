"""Check statistical and indexing assumptions of the Appendix F figures."""

import unittest

import numpy as np

from examples.ch26_visual_generation.open_model_figures import (
    checkpoint_level, displayed_rgb, forward_chain,
)
from examples.ch26_visual_generation.score_figure import gaussian_density_and_score


class OpenDiffusionFigureTests(unittest.TestCase):
    def test_forward_increments_have_the_prescribed_independent_variance(self):
        captures = forward_chain(np.zeros(200_000), [.2, .2], 7, {0, 1, 2})
        self.assertAlmostEqual(np.var(captures[1]), .2, delta=.003)
        self.assertAlmostEqual(np.var(captures[2]), .36, delta=.004)
        first_noise = captures[1] / np.sqrt(.2)
        second_noise = (captures[2] - np.sqrt(.8) * captures[1]) / np.sqrt(.2)
        self.assertLess(abs(np.corrcoef(first_noise, second_noise)[0, 1]), .01)

    def test_a_repeated_seed_is_reproducible_and_does_not_modify_the_input(self):
        clean = np.arange(12).reshape(3, 2, 2) / 10
        before = clean.copy()
        first = forward_chain(clean, [.1, .2], 11, {0, 1, 2})
        second = forward_chain(clean, [.1, .2], 11, {0, 1, 2})
        for level in first:
            np.testing.assert_array_equal(first[level], second[level])
        np.testing.assert_array_equal(clean, before)

    def test_display_mapping_and_checkpoint_indices_keep_fixed_conventions(self):
        np.testing.assert_array_equal(displayed_rgb(np.array([-2, 0, 2]).reshape(3, 1, 1)),
                                      np.array([0, .5, 1]).reshape(1, 1, 3))
        self.assertEqual(checkpoint_level(0), 1)
        self.assertEqual(checkpoint_level(999), 1000)
        with self.assertRaises(ValueError):
            displayed_rgb(np.zeros((2, 2)))

    def test_gaussian_score_matches_a_numerical_log_density_derivative(self):
        x = np.array([0., 1., 2., 3.])
        _, score = gaussian_density_and_score(x)
        h = 1e-5
        plus, _ = gaussian_density_and_score(x + h)
        minus, _ = gaussian_density_and_score(x - h)
        np.testing.assert_allclose(score, (np.log(plus) - np.log(minus)) / (2 * h), rtol=0, atol=1e-9)
        np.testing.assert_array_equal(score[[0, 3]], [2, -1])


if __name__ == "__main__":
    unittest.main()
