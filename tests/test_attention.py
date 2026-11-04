"""Numerical references and attention invariants, using NumPy and unittest."""

import unittest

import numpy as np

from examples.ch04_attention.attention import (
    attention_backward,
    scaled_dot_product_attention,
    stable_softmax,
)
from examples.ch04_attention.run_examples import H, K, Q, V


class SoftmaxTests(unittest.TestCase):
    def test_nonuniform_rows_normalize_independently(self):
        probabilities = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
        actual = stable_softmax(np.log(probabilities))
        np.testing.assert_allclose(actual, probabilities, atol=1e-15, rtol=0)
        np.testing.assert_allclose(actual.sum(axis=-1), [1.0, 1.0])

    def test_large_finite_scores(self):
        scores = np.array([[10000.0, 10001.0, 9999.0],
                           [-10000.0, -9999.0, -10001.0]])
        expected = np.array([0.2447284710547976, 0.6652409557748218,
                             0.0900305731703805])
        np.testing.assert_allclose(stable_softmax(scores), [expected, expected])
        with np.errstate(all="raise"):
            extreme = stable_softmax([[1e308, -1e308], [-1e308, -1e308]])
        np.testing.assert_array_equal(extreme, [[1.0, 0.0], [0.5, 0.5]])

    def test_mask_zeros_and_broadcasting(self):
        scores = np.zeros((2, 3, 4))
        mask = np.array([True, False, True, False])
        weights = stable_softmax(scores, mask)
        np.testing.assert_array_equal(weights[..., 1::2], 0.0)
        np.testing.assert_array_equal(weights[..., ::2], 0.5)
        np.testing.assert_array_equal(weights.sum(axis=-1), 1.0)

    def test_mask_shape_type_and_empty_rows(self):
        scores = np.ones((2, 3))
        with self.assertRaisesRegex(TypeError, "boolean"):
            stable_softmax(scores, np.ones((2, 3), dtype=int))
        for mask in (np.ones((4, 3), dtype=bool),
                     np.ones((1, 2, 3), dtype=bool)):
            with self.subTest(mask_shape=mask.shape):
                with self.assertRaisesRegex(ValueError, "cannot broadcast"):
                    stable_softmax(scores, mask)
        with self.assertRaisesRegex(ValueError, "at least one key"):
            stable_softmax(scores, [[True, False, False], [False, False, False]])

    def test_invalid_scores(self):
        for scores in (3.0, [], [[np.nan, 0.0]], [[np.inf, 0.0]]):
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    stable_softmax(scores)


class AttentionTests(unittest.TestCase):
    def test_three_token_projection_and_numeric_reference(self):
        np.testing.assert_array_equal(H, [[1, 0, 1, 0], [0, 1, 0, 2], [1, 1, 1, 1]])
        np.testing.assert_array_equal(Q, [[1, 0], [0, 1], [1, 1]])
        np.testing.assert_array_equal(K, Q)
        np.testing.assert_array_equal(V, [[1, 0], [0, 2], [1, 1]])
        expected_weights = np.array([
            [0.4011120926797859, 0.1977758146404282, 0.4011120926797859],
            [0.1977758146404282, 0.4011120926797859, 0.4011120926797859],
            [0.2482550782577231, 0.2482550782577231, 0.5034898434845538],
        ])
        expected_output = np.array([
            [0.8022241853595719, 0.7966637219606423],
            [0.5988879073202141, 1.2033362780393577],
            [0.7517449217422769, 1.0],
        ])
        output, weights = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(weights, expected_weights, atol=1e-15, rtol=0)
        np.testing.assert_allclose(output, expected_output, atol=1e-15, rtol=0)
        self.assertEqual(output.dtype, np.float64)

    def test_causal_prefix_and_future_perturbation(self):
        rng = np.random.default_rng(21)
        q, k = rng.normal(size=(2, 5, 4))
        v = rng.normal(size=(5, 3))
        allowed = np.tril(np.ones((5, 5), dtype=bool))
        output, weights = scaled_dot_product_attention(q, k, v, allowed)
        np.testing.assert_array_equal(weights[~allowed], 0.0)
        prefix, _ = scaled_dot_product_attention(
            q[:3], k[:3], v[:3], allowed[:3, :3]
        )
        np.testing.assert_allclose(prefix, output[:3], atol=1e-14, rtol=0)
        changed_q, changed_k, changed_v = q.copy(), k.copy(), v.copy()
        changed_q[3:] += 50.0
        changed_k[3:] -= 100.0
        changed_v[3:] *= -1000.0
        changed, _ = scaled_dot_product_attention(
            changed_q, changed_k, changed_v, allowed
        )
        np.testing.assert_allclose(changed[:3], output[:3], atol=1e-14, rtol=0)

    def test_unmasked_permutation_equivariance(self):
        rng = np.random.default_rng(8)
        q, k, v = rng.normal(size=(3, 5, 3))
        output, weights = scaled_dot_product_attention(q, k, v)
        p = np.array([3, 0, 4, 1, 2])
        permuted_output, permuted_weights = scaled_dot_product_attention(q[p], k[p], v[p])
        np.testing.assert_allclose(permuted_output, output[p], atol=1e-14, rtol=0)
        np.testing.assert_allclose(permuted_weights, weights[p][:, p], atol=1e-14, rtol=0)

    def test_batch_and_head_isolation(self):
        rng = np.random.default_rng(12)
        q = rng.normal(size=(2, 3, 4, 2))
        k = rng.normal(size=(2, 3, 5, 2))
        v = rng.normal(size=(2, 3, 5, 3))
        output, weights = scaled_dot_product_attention(q, k, v)
        self.assertEqual(output.shape, (2, 3, 4, 3))
        self.assertEqual(weights.shape, (2, 3, 4, 5))
        for batch in range(2):
            for head in range(3):
                separate, separate_weights = scaled_dot_product_attention(
                    q[batch, head], k[batch, head], v[batch, head]
                )
                np.testing.assert_allclose(output[batch, head], separate)
                np.testing.assert_allclose(weights[batch, head], separate_weights)
        changed_v = v.copy()
        changed_v[1, 2] += 10.0
        changed, _ = scaled_dot_product_attention(q, k, changed_v)
        unchanged = np.ones((2, 3), dtype=bool)
        unchanged[1, 2] = False
        np.testing.assert_allclose(changed[unchanged], output[unchanged])
        np.testing.assert_allclose(changed[1, 2], output[1, 2] + 10.0)

    def test_shapes_are_explicit(self):
        cases = [
            (np.ones(2), K, V, "sequence and feature"),
            (Q, K[None], V[None], "identical leading axes"),
            (np.ones((2, 1, 3, 2)), np.ones((1, 4, 3, 2)),
             np.ones((2, 4, 3, 2)), "identical leading axes"),
            (np.ones((3, 4)), K, V, "feature dimension"),
            (Q, K, V[:2], "number of tokens"),
            (np.empty((3, 0)), np.empty((3, 0)), V, "nonempty"),
        ]
        for q, k, v, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    scaled_dot_product_attention(q, k, v)
        with self.assertRaisesRegex(ValueError, "at least one key"):
            scaled_dot_product_attention(Q, K, V, np.zeros((3, 3), dtype=bool))


class BackwardTests(unittest.TestCase):
    def assert_finite_differences(self, q, k, v, upstream, allowed, coordinates):
        gradients = attention_backward(q, k, v, upstream, allowed)
        inputs = [q.copy(), k.copy(), v.copy()]
        step = 1e-6
        for argument, indices in enumerate(coordinates):
            self.assertEqual(gradients[argument].shape, inputs[argument].shape)
            for index in indices:
                with self.subTest(argument=argument, index=index, masked=allowed is not None):
                    original = inputs[argument][index]
                    inputs[argument][index] = original + step
                    plus, _ = scaled_dot_product_attention(*inputs, allowed)
                    loss_plus = np.sum(plus * upstream)
                    inputs[argument][index] = original - step
                    minus, _ = scaled_dot_product_attention(*inputs, allowed)
                    loss_minus = np.sum(minus * upstream)
                    inputs[argument][index] = original
                    numerical = (loss_plus - loss_minus) / (2.0 * step)
                    self.assertAlmostEqual(gradients[argument][index], numerical, delta=5e-10)

    def test_central_differences_unmasked_and_masked(self):
        rng = np.random.default_rng(34)
        q = rng.normal(size=(2, 3)) * 0.3
        k = rng.normal(size=(3, 3)) * 0.3
        v = rng.normal(size=(3, 2)) * 0.3
        upstream = rng.normal(size=(2, 2))
        coordinates = ([(0, 0), (1, 2)], [(0, 1), (1, 0), (2, 2)],
                       [(0, 1), (1, 0), (2, 1)])
        mask = np.array([[True, False, True], [True, True, False]])
        for allowed in (None, mask):
            self.assert_finite_differences(q, k, v, upstream, allowed, coordinates)

    def test_central_differences_with_batch_and_heads(self):
        rng = np.random.default_rng(72)
        q = rng.normal(size=(2, 2, 2, 3)) * 0.3
        k = rng.normal(size=(2, 2, 3, 3)) * 0.3
        v = rng.normal(size=(2, 2, 3, 2)) * 0.3
        upstream = rng.normal(size=(2, 2, 2, 2))
        mask = np.array([[True, False, True], [True, True, False]])
        coordinates = ([(0, 1, 0, 2), (1, 0, 1, 1)],
                       [(0, 0, 2, 1), (1, 1, 1, 0)],
                       [(0, 1, 0, 1), (1, 0, 2, 0)])
        self.assert_finite_differences(q, k, v, upstream, mask, coordinates)

    def test_globally_masked_key_has_zero_gradient(self):
        allowed = np.array([[True, False, True]] * 3)
        _, d_k, d_v = attention_backward(Q, K, V, np.ones((3, 2)), allowed)
        np.testing.assert_array_equal(d_k[1], 0.0)
        np.testing.assert_array_equal(d_v[1], 0.0)

    def test_upstream_shape_must_match_output(self):
        with self.assertRaisesRegex(ValueError, "output shape"):
            attention_backward(Q, K, V, np.ones((1, 3, 2)))


if __name__ == "__main__":
    unittest.main()
