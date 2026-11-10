"""Check Chapter 14's selective updates and SSD forms."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from examples.ch14_selective_ssm.model import (
    causal_decay,
    compose_diagonal,
    selective_scan,
    ssd_chunked,
    ssd_dense,
    ssd_recurrent,
    ssd_head_chunked,
    ssd_head_recurrent,
    zoh_coefficients,
)
from examples.ch14_selective_ssm.run_examples import (
    selective_example,
    ssd_example,
    ssd_inputs,
    warm_ssd_example,
    head_example,
)


class SelectiveScanTests(unittest.TestCase):
    def test_worked_scalar_example(self):
        report = selective_example()
        np.testing.assert_allclose(report["retention"], [0.5, 0.99, 0.5, 0.99])
        np.testing.assert_allclose(report["write_coefficients"], [0.5, 0.01, 0.5, 0.01])
        np.testing.assert_allclose(report["states"], [1, 1.08, 2.54, 2.5846])
        np.testing.assert_allclose(report["first_two_combined"], [0.495, 1.08])

    def test_zoh_preserves_tiny_write_and_supports_large_steps(self):
        with np.errstate(all="raise"):
            a, b = zoh_coefficients([1e-20, 1e3])
        np.testing.assert_array_equal(a, [1.0, 0.0])
        np.testing.assert_array_equal(b, [1e-20, 1.0])
        for bad in (0, -1, [np.inf], [np.nan], np.array([1j])):
            with self.subTest(step=bad), self.assertRaises(ValueError):
                zoh_coefficients(bad)

    def test_coordinatewise_retention_with_an_initial_state(self):
        a = [[0.5, 0.25], [0.25, 1], [0, 0.5]]
        b = [[1, 0.5], [2, -1], [1, 2]]
        c = [[1, 0], [0, 1], [1, -1]]
        x0 = np.array([4, 2], dtype=float)
        y, states = selective_scan([2, -1, 3], a, b, c, x0)
        np.testing.assert_array_equal(states, [[4, 1.5], [-1, 2.5], [3, 7.25]])
        np.testing.assert_array_equal(y, [4, 2.5, -4.25])
        np.testing.assert_array_equal(x0, [4, 2])

    def test_affine_composition_matches_order_and_grouping(self):
        first = (np.array([0.5, 0.25]), np.array([1, 3]))
        second = (np.array([0.25, 0.5]), np.array([2, -1]))
        third = (np.array([0.75, 0]), np.array([-1, 2]))
        x = np.array([3, -4])
        expected = third[0] * (second[0] * (first[0] * x + first[1]) + second[1]) + third[1]
        left = compose_diagonal(compose_diagonal(first, second), third)
        right = compose_diagonal(first, compose_diagonal(second, third))
        np.testing.assert_array_equal(left[0] * x + left[1], expected)
        np.testing.assert_array_equal(left, right)
        with self.assertRaisesRegex(ValueError, "share a shape"):
            compose_diagonal(first, (0.5, 1))

    def test_scan_checks_shapes_and_retention(self):
        u, b, c = [1, 2], [[1], [1]], [[1], [1]]
        with self.assertRaisesRegex(ValueError, "per time step"):
            selective_scan(u, [[0.5]], b, c)
        with self.assertRaisesRegex(ValueError, "match u"):
            selective_scan(u, [[0.5], [0.5]], b, [[1, 1], [1, 1]])
        with self.assertRaisesRegex(ValueError, "state width"):
            selective_scan(u, [[0.5], [0.5]], b, c, x0=[0, 0])
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            selective_scan(u, [[0.5], [1.1]], b, c)


class SSDTests(unittest.TestCase):
    def test_worked_ssd_matrix_and_chunk_boundary(self):
        report = ssd_example()
        np.testing.assert_array_equal(report["matrix"], [
            [1, 0, 0, 0], [0.5, 1, 0, 0], [0, 0.5, 1, 0], [0.125, 0.25, 1, 1],
        ])
        for method in ("dense", "recurrent", "chunked"):
            np.testing.assert_array_equal(report[method], [2, 1, 4, 4.25])
        np.testing.assert_array_equal(report["second_chunk_within"], [4, 4])
        np.testing.assert_array_equal(report["second_chunk_from_incoming_state"], [0, 0.25])
        np.testing.assert_array_equal(report["chunked_final_state"], [2.25, 2])

    def test_short_final_chunk_with_varying_retention_and_initial_state(self):
        report = warm_ssd_example()
        self.assertEqual(report["chunk_lengths"], [2, 2, 1])
        for method in ("dense", "recurrent", "chunked"):
            np.testing.assert_array_equal(report[method], [3, 0.625, 3.9375, 0, 3])
        np.testing.assert_array_equal(report["chunked_final_state"], [2, -1])

    def test_causal_products_exclude_source_retention_and_allow_zero(self):
        a = [0, 0.5, 0, 0.25]
        np.testing.assert_array_equal(causal_decay(a), [
            [1, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0.25, 1],
        ])

    def test_local_blocks_use_only_the_chosen_chunk_size(self):
        u, a, b, c = ssd_inputs()
        with patch(
            "examples.ch14_selective_ssm.model.causal_decay", wraps=causal_decay
        ) as build_decay:
            ssd_chunked(u, a, b, c, chunk_size=3)
        self.assertEqual([len(call.args[0]) for call in build_decay.call_args_list], [3, 1])

    def test_chunk_sizes_against_explicit_causal_products(self):
        rng = np.random.default_rng(14)
        u = rng.normal(size=7)
        a = np.array([0.3, 0.7, 0.2, 0, 0.9, 1, 0.4])
        b, c = rng.normal(size=(2, len(u), 3))
        x0 = rng.normal(size=3)
        matrix = np.array([
            [float(c[t] @ b[s]) * math.prod(a[s + 1:t + 1]) if s <= t else 0.0
             for s in range(len(u))]
            for t in range(len(u))
        ])
        expected = matrix @ u + np.array([
            math.prod(a[:t + 1]) * (c[t] @ x0) for t in range(len(u))
        ])
        dense, got_matrix = ssd_dense(u, a, b, c, x0)
        recurrent, states = ssd_recurrent(u, a, b, c, x0)
        np.testing.assert_allclose(got_matrix, matrix, rtol=0, atol=1e-14)
        np.testing.assert_allclose(dense, expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(recurrent, expected, rtol=0, atol=1e-14)
        for size in (1, 2, 3, 4, 7, 10):
            with self.subTest(size=size):
                chunked, last = ssd_chunked(u, a, b, c, size, x0)
                np.testing.assert_allclose(chunked, expected, rtol=0, atol=1e-14)
                np.testing.assert_allclose(last, states[-1], rtol=0, atol=1e-14)

    def test_future_coefficients_and_inputs_leave_earlier_outputs_alone(self):
        u, a, b, c = ssd_inputs()
        for method in (ssd_recurrent, ssd_dense,
                       lambda *args: ssd_chunked(*args, chunk_size=3)):
            y, _ = method(u, a, b, c)
            u2, a2, b2, c2 = [arr.copy() for arr in (u, a, b, c)]
            u2[2:] += 5
            a2[2:] = 1
            b2[2:] += 2
            c2[2:] -= 3
            changed, _ = method(u2, a2, b2, c2)
            np.testing.assert_array_equal(changed[:2], y[:2])
            self.assertGreater(np.max(np.abs(changed[2:] - y[2:])), 1)

    def test_ssd_checks_retention_length_and_chunk_size(self):
        u, a, b, c = ssd_inputs()
        with self.assertRaisesRegex(ValueError, "one retention"):
            ssd_dense(u, a[:-1], b, c)
        with self.assertRaisesRegex(ValueError, "chunk size"):
            ssd_chunked(u, a, b, c, 0)


class SSDHeadTests(unittest.TestCase):
    def test_outer_product_write_and_read_with_rectangular_state(self):
        report = head_example()
        np.testing.assert_array_equal(report["outputs"], [[2, -1], [1, 1.5], [4, 1], [4.25, 1.375]])
        np.testing.assert_array_equal(report["final_state"], [[2.25, .875], [2, .5]])
        u = np.array([[2, -1, 3], [0, 2, 1], [1, 0, -2]], dtype=float)
        b = np.array([[1, 2], [-1, 1], [2, 0]], dtype=float)
        c = np.array([[1, 0], [0, 1], [1, -1]], dtype=float)
        initial = np.array([[2, 0, 1], [0, 1, -1]], dtype=float)
        y, states = ssd_head_recurrent(u, [0.5, 0.25, 0], b, c, initial)
        np.testing.assert_array_equal(states[0], [[3, -1, 3.5], [4, -1.5, 5.5]])
        np.testing.assert_array_equal(states[1], [[0.75, -2.25, -0.125], [1, 1.625, 2.375]])
        np.testing.assert_array_equal(states[2], [[2, 0, -4], [0, 0, 0]])
        np.testing.assert_array_equal(y, [[3, -1, 3.5], [1, 1.625, 2.375], [2, 0, -4]])
        np.testing.assert_array_equal(initial, [[2, 0, 1], [0, 1, -1]])

    def test_all_features_against_independent_dense_products(self):
        rng = np.random.default_rng(1402)
        u = rng.normal(size=(11, 4))
        b, c = rng.normal(size=(2, 11, 3))
        a = np.array([.1, .9, .7, 0, .4, 1, .2, .6, .8, .3, .9])
        initial = rng.normal(size=(3, 4))
        matrix = np.array([[c[t] @ b[s] * math.prod(a[s+1:t+1]) if s <= t else 0
                            for s in range(11)] for t in range(11)])
        expected = matrix @ u + np.array([math.prod(a[:t+1]) * (c[t] @ initial)
                                          for t in range(11)])
        recurrent, states = ssd_head_recurrent(u, a, b, c, initial)
        np.testing.assert_allclose(recurrent, expected, atol=1e-14, rtol=0)
        for size in (1, 2, 4, 7, 11, 16):
            with self.subTest(chunk=size):
                chunked, final = ssd_head_chunked(u, a, b, c, size, initial)
                np.testing.assert_allclose(chunked, expected, atol=1e-14, rtol=0)
                np.testing.assert_allclose(final, states[-1], atol=1e-14, rtol=0)

    def test_saved_head_state_continues_stream_and_future_changes_are_causal(self):
        rng = np.random.default_rng(1403)
        u = rng.normal(size=(7, 3))
        b, c = rng.normal(size=(2, 7, 2))
        a = np.full(7, .8)
        y, state = ssd_head_chunked(u, a, b, c, 3)
        first, boundary = ssd_head_chunked(u[:4], a[:4], b[:4], c[:4], 3)
        second, final = ssd_head_chunked(u[4:], a[4:], b[4:], c[4:], 2, boundary)
        np.testing.assert_allclose(np.vstack([first, second]), y, atol=1e-14, rtol=0)
        np.testing.assert_allclose(final, state, atol=1e-14, rtol=0)
        changed = u.copy()
        changed[4:] += 20
        after, _ = ssd_head_chunked(changed, a, b, c, 3)
        np.testing.assert_array_equal(after[:4], y[:4])
        self.assertGreater(np.max(np.abs(after[4:] - y[4:])), 1)

    def test_head_checks_axis_shapes_and_chunk_size(self):
        u = np.ones((4, 3)); b = c = np.ones((4, 2)); a = np.full(4, .5)
        with self.assertRaisesRegex(ValueError, "state, feature"):
            ssd_head_chunked(u, a, b, c, 2, np.zeros((3, 2)))
        with self.assertRaisesRegex(ValueError, "match u"):
            ssd_head_recurrent(u, a, b[:3], c[:3])
        with self.assertRaisesRegex(ValueError, "chunk size"):
            ssd_head_chunked(u, a, b, c, 0)


if __name__ == "__main__":
    unittest.main()
