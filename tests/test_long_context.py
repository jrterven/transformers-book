"""Check the Chapter 11 connectivity and cache-read examples."""

import unittest

import numpy as np

from examples.ch04_attention.attention import scaled_dot_product_attention
from examples.ch11_long_context.model import (
    block_summaries,
    local_mask,
    selected_attention,
)
from examples.ch11_long_context.run_examples import (
    compression_example,
    mixed_history_example,
    selection_example,
)


class LocalMaskTests(unittest.TestCase):
    def test_local_and_global_connectivity(self):
        local = local_mask(8, 1)
        with_global = local_mask(8, 1, (0,))
        np.testing.assert_array_equal(local.sum(axis=1), [2, 3, 3, 3, 3, 3, 3, 2])
        self.assertEqual(local.sum(), 22)
        self.assertEqual(with_global.sum(), 34)
        self.assertTrue(with_global[0].all() and with_global[:, 0].all())

    def test_causal_boundary_also_applies_to_global_edges(self):
        np.testing.assert_array_equal(local_mask(4, 0, (1,), causal=True), [
            [True, False, False, False],
            [True, True, False, False],
            [False, True, True, False],
            [False, True, False, True],
        ])

    def test_rejects_invalid_positions_and_radius(self):
        for positions in ((-1,), (4,)):
            with self.subTest(positions=positions):
                with self.assertRaisesRegex(ValueError, "outside"):
                    local_mask(4, 1, positions)
        with self.assertRaisesRegex(ValueError, "radius"):
            local_mask(4, -1)


class SelectionTests(unittest.TestCase):
    def test_worked_example_and_causal_ranking(self):
        result = selection_example()
        self.assertEqual(result["selected"], [1, 6])
        np.testing.assert_allclose(result["main_scores"], [np.log(3), 0])
        np.testing.assert_allclose(result["main_weights"], [0.75, 0.25])
        np.testing.assert_allclose(result["output"], [0.75, 0.5])
        self.assertEqual(result["causal_selected_with_large_future_score"], [1, 6])

    def test_all_allowed_entries_match_masked_dense_attention(self):
        query = np.array([1.0, -0.5])
        keys = np.array([[1, 0], [0, 2], [1, 1], [-1, 0]])
        values = np.arange(12).reshape(4, 3)
        allowed = np.array([True, False, True, True])
        output, selected, weights = selected_attention(
            query, keys, values, [3, 99, -2, 7], 9, allowed
        )
        dense, dense_weights = scaled_dot_product_attention(
            query[None, :], keys, values, allowed
        )
        np.testing.assert_array_equal(selected, [3, 0, 2])
        np.testing.assert_allclose(output, dense[0], rtol=0, atol=1e-14)
        np.testing.assert_allclose(weights, dense_weights[0, selected], rtol=0, atol=1e-14)

    def test_ties_and_unselected_entries(self):
        query = [0]
        keys = [[np.nan], [0], [0], [np.nan]]
        values = [[np.nan], [5], [6], [np.nan]]
        output, selected, weights = selected_attention(
            query, keys, values, [np.inf, 4, 4, 1], 2,
            allowed=[False, True, True, True],
        )
        np.testing.assert_array_equal(selected, [1, 2])
        np.testing.assert_array_equal(weights, [0.5, 0.5])
        np.testing.assert_array_equal(output, [5.5])

    def test_invalid_candidates_and_mask(self):
        q, k, v = [0], [[0], [0]], [[1], [2]]
        with self.assertRaisesRegex(ValueError, "candidate"):
            selected_attention(q, k, v, [1, 2], 1, [False, False])
        with self.assertRaisesRegex(ValueError, "finite"):
            selected_attention(q, k, v, [1, np.inf], 1)
        with self.assertRaisesRegex(ValueError, "boolean"):
            selected_attention(q, k, v, [1, 2], 1, [1, 0])
        with self.assertRaisesRegex(ValueError, "k must be positive"):
            selected_attention(q, k, v, [1, 2], 0)

    def test_rejects_complex_projected_arrays_and_index_scores(self):
        inputs = ([1.0], [[1.0], [0.0]], [[2.0], [4.0]], [1.0, 0.0])
        for i in range(len(inputs)):
            changed = list(inputs)
            changed[i] = np.asarray(changed[i], dtype=complex) + 3j
            with self.subTest(input_index=i):
                with self.assertRaisesRegex(ValueError, "real numbers"):
                    selected_attention(*changed, 1)

    def test_integer_scores_keep_exact_ranking_and_stable_ties(self):
        large = 2**53
        for dtype in (np.int64, np.uint64):
            scores = np.array([large, large + 1, large + 1], dtype=dtype)
            with self.subTest(dtype=dtype):
                _, selected, _ = selected_attention(
                    [0], [[0], [0], [0]], [[2], [4], [8]], scores, 2
                )
                np.testing.assert_array_equal(selected, [1, 2])


class SummaryTests(unittest.TestCase):
    def test_rejects_complex_features_and_gates(self):
        features, gates = np.array([[1.0], [3.0]]), np.zeros((2, 1))
        for inputs in ((features + 3j, gates), (features, gates + 3j)):
            with self.assertRaisesRegex(ValueError, "real numbers"):
                block_summaries(*inputs, 2)

    def test_softmax_is_over_time_separately_for_each_coordinate(self):
        features = np.array([[1, 5], [3, 5], [2, 1], [6, 1]])
        logits = np.log([[0.1, 0.4], [0.2, 0.3], [0.3, 0.2], [0.4, 0.1]])
        summaries, ends = block_summaries(features, logits, 4)
        np.testing.assert_array_equal(ends, [3])
        np.testing.assert_allclose(summaries, [[3.7, 3.8]])
        shifted, _ = block_summaries(features, logits + [1e4, -1e4], 4)
        np.testing.assert_allclose(shifted, summaries, rtol=0, atol=1e-11)

    def test_overlap_truncates_first_window_and_leaves_partial_tail_raw(self):
        features = np.arange(1.0, 8.0)[:, None]
        gates = np.zeros_like(features)
        overlap, ends = block_summaries(features, gates, 2, overlap=True)
        disjoint, disjoint_ends = block_summaries(features, gates, 2)
        np.testing.assert_array_equal(ends, [1, 3, 5])
        np.testing.assert_array_equal(disjoint_ends, ends)
        np.testing.assert_array_equal(overlap[:, 0], [1.5, 2.5, 4.5])
        np.testing.assert_array_equal(disjoint[:, 0], [1.5, 3.5, 5.5])
        empty, no_ends = block_summaries(features[:1], gates[:1], 2)
        self.assertEqual(empty.shape, (0, 1))
        self.assertEqual(no_ends.size, 0)

    def test_window_counts(self):
        record = compression_example()
        self.assertAlmostEqual(record["gated_scalar_summary"], 3.7)
        self.assertEqual(record["stride_2_overlap_ends"], list(range(1, 16, 2)))
        self.assertEqual(record["stride_4_disjoint_ends"], [3, 7, 11, 15])
        self.assertEqual(record["raw_recent_positions"], [12, 13, 14, 15])

    def test_mixed_history_uses_only_finished_summaries_and_one_softmax(self):
        record = mixed_history_example()
        self.assertEqual(record["summary_ends"], [1, 3, 5, 7])
        self.assertEqual(record["visible_summary_ends"], [1, 3, 5])
        self.assertEqual(record["selected_summary"], 0)
        self.assertEqual(record["raw_recent_positions"], [5, 6])
        self.assertEqual(record["gathered_values"], [2, 5, 8])
        np.testing.assert_allclose(record["main_weights"], [1 / 3] * 3)
        self.assertAlmostEqual(record["output"], 5)


if __name__ == "__main__":
    unittest.main()
