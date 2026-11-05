"""Check Chapter 13's recurrence, convolutions, and gates."""

import unittest

import numpy as np

from examples.ch13_state_spaces.model import (
    bilinear_discretize,
    causal_convolution,
    fft_causal_convolution,
    hyena_stage,
    ssm_kernel,
    ssm_scan,
)
from examples.ch13_state_spaces.run_examples import (
    hyena_example,
    padding_example,
    scalar_example,
    vector_example,
)


class StateSpaceTests(unittest.TestCase):
    def test_worked_scalar_system(self):
        result = scalar_example()
        np.testing.assert_array_equal(result["kernel"], [1, 0.5, 0.25, 0.125])
        expected = [2, 1, 4.5, 2.25]
        np.testing.assert_array_equal(result["scan"], expected)
        np.testing.assert_array_equal(result["direct_convolution"], expected)
        np.testing.assert_allclose(result["fft_convolution"], expected, rtol=0, atol=1e-14)

    def test_coupled_vector_state(self):
        result = vector_example()
        np.testing.assert_array_equal(result["states"], [
            [2, 4], [2, 1], [5.25, 8.25], [4.6875, 2.0625],
        ])
        np.testing.assert_array_equal(result["scan"], [-2, 1, -3, 2.625])
        np.testing.assert_array_equal(result["kernel"], [-1, 0.5, 0.5, 0.3125])
        self.assertEqual(result["direct_max_error"], 0)
        self.assertLess(result["fft_max_error"], 1e-14)

    def test_initial_state_adds_homogeneous_response(self):
        u = [2, 0, 4, 0]
        a = np.array([[0.5, 0.25], [0, 0.25]])
        b, c = np.array([1, 2]), np.array([1, -1])
        x0 = np.array([2, -1])
        kernel = ssm_kernel(a, b, c, len(u))
        from_inputs = causal_convolution(u, kernel)
        from_x0 = np.array([c @ np.linalg.matrix_power(a, t) @ x0
                            for t in range(1, len(u) + 1)])
        outputs, _ = ssm_scan(u, a, b, c, x0=x0)
        np.testing.assert_array_equal(outputs, from_inputs + from_x0)
        np.testing.assert_array_equal(x0, [2, -1])

    def test_kernel_matches_matrix_power_and_impulse_scan(self):
        a = np.array([[0.3, -0.2], [0.4, 0.1]])
        b, c = np.array([0.8, -0.6]), np.array([1.0, 2.0])
        inputs = [1] + [0] * 8
        expected = [c @ np.linalg.matrix_power(a, lag) @ b
                    for lag in range(len(inputs))]
        kernel = ssm_kernel(a, b, c, len(inputs))
        scan, _ = ssm_scan(inputs, a, b, c)
        np.testing.assert_allclose(kernel, expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(kernel, scan, rtol=0, atol=1e-14)

    def test_checks_state_shapes_and_real_inputs(self):
        with self.assertRaisesRegex(ValueError, "width matching B"):
            ssm_scan([1], [[1, 0]], [1], [1])
        with self.assertRaisesRegex(ValueError, "B and C"):
            ssm_kernel([[1]], [1], [1, 2], 3)
        with self.assertRaisesRegex(ValueError, "x0"):
            ssm_scan([1], [[1]], [1], [1], x0=[0, 0])
        with self.assertRaisesRegex(ValueError, "real numbers"):
            ssm_scan(np.array([1j]), [[1]], [1], [1])
        with self.assertRaisesRegex(ValueError, "positive"):
            ssm_kernel([[1]], [1], [1], 0)


class DiscretizationTests(unittest.TestCase):
    def test_scalar_bilinear_and_two_solve_residuals(self):
        a_bar, b_bar = bilinear_discretize([[-2]], [1], 0.5)
        np.testing.assert_allclose(a_bar, [[1 / 3]])
        np.testing.assert_allclose(b_bar, [1 / 3])
        a = np.array([[-2.0, 0.5], [-0.25, -0.75]])
        b = np.array([1.0, -1.0])
        delta = 0.2
        a_bar, b_bar = bilinear_discretize(a, b, delta)
        left = np.eye(2) - (delta / 2) * a
        np.testing.assert_allclose(left @ a_bar, np.eye(2) + (delta / 2) * a)
        np.testing.assert_allclose(left @ b_bar, delta * b)

    def test_invalid_step_singular_system_and_overflow(self):
        for delta in (0, -0.1, np.nan, np.inf, [0.5], np.array(0.5 + 1j)):
            with self.subTest(delta=delta):
                with self.assertRaisesRegex(ValueError, "delta"):
                    bilinear_discretize([[-2]], [1], delta)
        with self.assertRaises(np.linalg.LinAlgError):
            bilinear_discretize([[2]], [1], 1)
        with self.assertRaises(FloatingPointError):
            bilinear_discretize([[0]], [1e308], 2)


class ConvolutionTests(unittest.TestCase):
    def test_fft_matches_direct_for_odd_even_and_unequal_lengths(self):
        rng = np.random.default_rng(13)
        for input_length, kernel_length in ((1, 1), (2, 8), (3, 3), (8, 3), (9, 8)):
            with self.subTest(input_length=input_length, kernel_length=kernel_length):
                u = rng.normal(size=input_length)
                kernel = rng.normal(size=kernel_length)
                expected = np.convolve(u, kernel, mode="full")[:input_length]
                np.testing.assert_allclose(
                    causal_convolution(u, kernel), expected, rtol=0, atol=1e-14
                )
                np.testing.assert_allclose(
                    fft_causal_convolution(u, kernel), expected, rtol=0, atol=1e-14
                )

    def test_linear_padding_prevents_wraparound(self):
        result = padding_example()
        self.assertEqual(result["fft_length"], 6)
        np.testing.assert_allclose(result["padded_fft"], [1, 1, 1, 2], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result["circular_at_input_length"], [3, 3, 1, 2])
        last_input = [0, 0, 0, 1]
        kernel = [1, 0.5, 0.25, 0.125]
        np.testing.assert_allclose(
            fft_causal_convolution(last_input, kernel), last_input, rtol=0, atol=1e-14
        )

    def test_future_inputs_cannot_change_earlier_outputs(self):
        a, b, c = [[0.5]], [1], [1]
        before, after = [2, 0, 4, 0], [2, 0, 4, 8]
        kernel = ssm_kernel(a, b, c, len(before))
        for method in (causal_convolution, fft_causal_convolution):
            expected = method(before, kernel)
            perturbed = method(after, kernel)
            np.testing.assert_allclose(perturbed[:-1], expected[:-1], rtol=0, atol=1e-14)
            self.assertAlmostEqual(perturbed[-1] - expected[-1], 8)

    def test_hyena_worked_gates_and_invalid_shape(self):
        result = hyena_example()
        np.testing.assert_array_equal(result["gated_input"], [1, 0, 0, 1])
        np.testing.assert_array_equal(result["filtered"], [1, 0.5, 0.25, 1.125])
        np.testing.assert_array_equal(result["output"], [1, 0.5, 0.5, 1.125])
        with self.assertRaisesRegex(ValueError, "each gate"):
            hyena_stage([1, 2], [1, 0], [2], [1])


if __name__ == "__main__":
    unittest.main()
