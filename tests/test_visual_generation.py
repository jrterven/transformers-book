"""Check Chapter 25's forward mix, guidance, and deterministic updates."""

import unittest

import numpy as np

from examples.ch25_visual_generation.model import add_noise, ddim_step, guided_noise
from examples.ch25_visual_generation.run_examples import denoising_example, guidance_example


class DiffusionTests(unittest.TestCase):
    def test_worked_forward_marginal_and_endpoints(self):
        clean = np.eye(2)
        noise = [[0, 1], [-1, 0]]
        np.testing.assert_array_equal(add_noise(clean, noise, 0.36), [[0.6, 0.8], [-0.8, 0.6]])
        np.testing.assert_array_equal(add_noise(clean, noise, 0), noise)
        np.testing.assert_array_equal(add_noise(clean, noise, 1), clean)

    def test_exact_prediction_matches_the_next_forward_marginal(self):
        clean = np.array([-2, 3.5])
        noise = np.array([2, -4])
        noisy = add_noise(clean, noise, 0.12)
        estimated, next_noisy = ddim_step(noisy, noise, 0.12, 0.7)
        np.testing.assert_allclose(estimated, clean, rtol=0, atol=1e-14)
        np.testing.assert_allclose(next_noisy, add_noise(clean, noise, 0.7), rtol=0, atol=1e-14)
        final_estimate, final = ddim_step(next_noisy, noise, 0.7, 1)
        np.testing.assert_array_equal(final, final_estimate)
        np.testing.assert_allclose(final, clean, rtol=0, atol=1e-14)

    def test_perturbed_epsilon_changes_one_coordinate_through_two_steps(self):
        result = denoising_example()
        exact = result["exact_prediction"]
        self.assertEqual(exact["final_max_absolute_error"], 0)
        perturbed = result["perturbed_prediction"]
        np.testing.assert_allclose(perturbed["estimated_clean_from_first_step"],
                                   [[2 / 3, 0], [0, 1]])
        np.testing.assert_allclose(perturbed["next_noisy"], [[41 / 60, 0.6], [-0.6, 0.8]])
        np.testing.assert_allclose(perturbed["final"], [[2 / 3, 0], [0, 1]])
        self.assertAlmostEqual(perturbed["final_max_absolute_error"], 1 / 3)

    def test_clip_shaped_arrays_stay_aligned_and_unmodified(self):
        clean = np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2) / 10
        noise = np.flip(clean, axis=0).copy() * -0.5
        before_clean, before_noise = clean.copy(), noise.copy()
        noisy = add_noise(clean, noise, np.float64(0.4))
        _, unchanged = ddim_step(noisy, noise, 0.4, 0.4)
        first = ddim_step(noisy, noise, 0.4, 1)
        repeated = ddim_step(noisy, noise, 0.4, 1)
        self.assertEqual(first[0].shape, clean.shape)
        self.assertEqual(first[0].dtype, np.float64)
        np.testing.assert_allclose(unchanged, noisy, rtol=0, atol=1e-14)
        np.testing.assert_allclose(first[1], clean, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(clean, before_clean)
        np.testing.assert_array_equal(noise, before_noise)

    def test_invalid_alphas_and_schedule(self):
        for alpha in (-0.01, 1.1):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(ValueError, "between"):
                add_noise([0], [1], alpha)
        with self.assertRaisesRegex(ValueError, "scalar"):
            add_noise([0], [1], [0.5])
        for current, next_alpha in ((0, 0.6), (0.6, 0.5), (0.6, 1.01)):
            with self.subTest(current=current, next_alpha=next_alpha):
                with self.assertRaisesRegex(ValueError, "alpha_bar_t"):
                    ddim_step([0], [1], current, next_alpha)

    def test_array_validation_excludes_broadcasts_and_nonfinite_values(self):
        for transform in (lambda a, b: add_noise(a, b, 0.5),
                          lambda a, b: guided_noise(a, b, 2),
                          lambda a, b: ddim_step(a, b, 0.5, 0.7)):
            with self.assertRaisesRegex(ValueError, "same shape"):
                transform([[1], [2]], [1, 2])
            with self.assertRaisesRegex(ValueError, "finite"):
                transform([np.inf], [1])
            with self.assertRaisesRegex(ValueError, "nonempty and real"):
                transform([1j], [1])
            with self.assertRaisesRegex(ValueError, "nonempty and real"):
                transform([], [])


class GuidanceTests(unittest.TestCase):
    def test_worked_guidance_endpoints_and_extrapolation(self):
        example = guidance_example()
        unconditional = example["unconditional"]
        conditional = example["conditional"]
        np.testing.assert_allclose(example["guided"], [1, 0.6])
        np.testing.assert_allclose(guided_noise(unconditional, conditional, 0), unconditional)
        np.testing.assert_allclose(guided_noise(unconditional, conditional, 1), conditional)
        np.testing.assert_allclose(guided_noise(unconditional, conditional, 0.5), [0.4, 0])

    def test_guidance_requires_a_finite_scalar(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            guided_noise([0], [1], np.nan)
        with self.assertRaisesRegex(ValueError, "scalar"):
            guided_noise([0], [1], [2])


if __name__ == "__main__":
    unittest.main()
