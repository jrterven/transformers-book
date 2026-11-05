"""Check Chapter 12's online recurrence and tiled attention."""

import unittest

import numpy as np

from examples.ch04_attention.attention import scaled_dot_product_attention
from examples.ch12_efficient.model import online_update, score_workspace, tiled_attention
from examples.ch12_efficient.run_examples import online_example, projected_example


class OnlineSoftmaxTests(unittest.TestCase):
    def test_worked_example_and_unchanged_masked_tile(self):
        example = online_example()
        first, masked, last = example["states"]
        self.assertEqual(masked, first)
        self.assertAlmostEqual(first["maximum"], np.log(2))
        self.assertAlmostEqual(first["denominator"], 1.5)
        self.assertAlmostEqual(first["numerator"], 2.5)
        self.assertAlmostEqual(last["maximum"], np.log(3))
        self.assertAlmostEqual(last["denominator"], 2)
        self.assertAlmostEqual(last["numerator"], 17 / 3)
        self.assertAlmostEqual(example["output"], 17 / 6)

    def test_empty_row_stays_empty_until_its_first_key(self):
        empty = np.full(2, -np.inf), np.zeros(2), np.zeros((2, 1))
        with np.errstate(all="raise"):
            maximum, denominator, numerator = online_update(
                *empty, [[999, 998], [0, np.log(2)]], [[1], [2]],
                [[False, False], [True, True]],
            )
            np.testing.assert_allclose(maximum, [-np.inf, np.log(2)])
            np.testing.assert_allclose(denominator, [0, 1.5])
            np.testing.assert_allclose(numerator, [[0], [2.5]])
            maximum, denominator, numerator = online_update(
                maximum, denominator, numerator, [[np.log(3)], [999]], [[4]],
                [[True], [False]],
            )
        np.testing.assert_allclose(maximum, [np.log(3), np.log(2)])
        np.testing.assert_allclose(denominator, [1, 1.5])
        np.testing.assert_allclose(numerator, [[4], [2.5]])

    def test_large_finite_scores_use_maximum_as_reference(self):
        empty = np.full(1, -np.inf), np.zeros(1), np.zeros((1, 1))
        with np.errstate(all="raise"):
            state = online_update(*empty, [[-1e308]], [[-2.0]])
            maximum, denominator, numerator = online_update(*state, [[1e308]], [[9.0]])
        np.testing.assert_array_equal(maximum, [1e308])
        np.testing.assert_array_equal(denominator, [1.0])
        np.testing.assert_array_equal(numerator, [[9.0]])

    def test_rejects_invalid_tile_mask(self):
        empty = np.full(1, -np.inf), np.zeros(1), np.zeros((1, 1))
        with self.assertRaisesRegex(ValueError, "boolean mask"):
            online_update(*empty, [[0]], [[1]], [[1]])
        with self.assertRaisesRegex(ValueError, "boolean mask"):
            online_update(*empty, [[0]], [[1]], [[True, True]])


class TiledAttentionTests(unittest.TestCase):
    def test_tiles_match_dense_for_rectangular_unmasked_and_causal_attention(self):
        q, k, v = projected_example()
        allowed = np.arange(len(k))[None, :] <= 2 + np.arange(len(q))[:, None]
        for mask in (None, allowed):
            expected, _ = scaled_dot_product_attention(q, k, v, mask)
            for query_tile, key_tile in ((1, 1), (2, 5), (3, 4), (20, 20)):
                with self.subTest(causal=mask is not None,
                                  query_tile=query_tile, key_tile=key_tile):
                    output = tiled_attention(
                        q, k, v, query_tile, key_tile,
                        causal=mask is not None, query_start=2,
                    )
                    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-13)

    def test_causal_queries_do_not_read_future_rows(self):
        q, k, v = projected_example()
        before = tiled_attention(q, k, v, 3, 4, causal=True, query_start=2)
        k[3:] += 20
        v[3:] += 100
        after = tiled_attention(q, k, v, 3, 4, causal=True, query_start=2)
        np.testing.assert_allclose(after[0], before[0], rtol=0, atol=1e-14)
        self.assertGreater(np.max(np.abs(after[1:] - before[1:])), 1)

    def test_float32_inputs_and_more_queries_than_keys(self):
        q, k, v = projected_example()
        q, k, v = q.astype(np.float32), k[:3].astype(np.float32), v[:3].astype(np.float32)
        expected, _ = scaled_dot_product_attention(q, k, v)
        output = tiled_attention(q, k, v, 4, 2)
        self.assertEqual(output.dtype, np.float64)
        np.testing.assert_allclose(output, expected, rtol=0, atol=1e-13)

    def test_rejects_invalid_projection_or_position(self):
        q, k, v = projected_example()
        with self.assertRaisesRegex(ValueError, "widths"):
            tiled_attention(q[:, :2], k, v, 2, 2)
        with self.assertRaisesRegex(ValueError, "lengths"):
            tiled_attention(q, k, v[:-1], 2, 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            tiled_attention(q, k, v, 0, 2)
        with self.assertRaisesRegex(ValueError, "key prefix"):
            tiled_attention(q, k, v, 2, 2, causal=True, query_start=3)
        with self.assertRaisesRegex(ValueError, "real numbers"):
            tiled_attention(q, k, v + 1j, 2, 2)
        with self.assertRaisesRegex(ValueError, "finite"):
            tiled_attention(q, k, v * np.nan, 2, 2)


class WorkspaceTests(unittest.TestCase):
    def test_scores_and_tiles_count_only_one_array(self):
        self.assertEqual(score_workspace(7, 9, 3, 4), (63, 12))
        self.assertEqual(score_workspace(7, 9, 20, 20), (63, 63))
        self.assertEqual(score_workspace(4096, 4096, 64, 128), (4096**2, 8192))
        n = np.int64(5_000_000_000)
        self.assertEqual(score_workspace(n, n, 8, 16), (25_000_000_000_000_000_000, 128))


if __name__ == "__main__":
    unittest.main()
