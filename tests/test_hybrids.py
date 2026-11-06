"""Check Chapter 15's delta update, attention read, and storage counts."""

import unittest

import numpy as np

from examples.ch04_attention.attention import scaled_dot_product_attention
from examples.ch15_hybrids.model import (
    attention_token,
    delta_step,
    history_counts,
    initial_stores,
    stack_step,
)
from examples.ch15_hybrids.run_examples import HYBRID, delta_example, schedule_example, toy_tokens


class DeltaTests(unittest.TestCase):
    def test_worked_delta_update(self):
        result = delta_example()
        np.testing.assert_array_equal(result["decayed"], [[1, 0], [0.5, 1.5]])
        np.testing.assert_array_equal(result["prediction"], [1, 0.5])
        np.testing.assert_array_equal(result["error"], [3, 0.5])
        np.testing.assert_array_equal(result["updated_state"], [[2.5, 0], [0.75, 1.5]])
        np.testing.assert_array_equal(result["key_read"], [2.5, 0.75])
        np.testing.assert_array_equal(result["orthogonal_read"], [0, 1.5])

    def test_unit_key_negative_and_orthogonal_carry(self):
        state = np.array([[1.0, 2.0], [-3.0, 4.0], [2.0, 0.0]])
        key = np.array([0.6, 0.8])
        orthogonal = np.array([-0.8, 0.6])
        y, updated = delta_step(state, key, key, [0, 0, 0], 0.8, 1.5)
        np.testing.assert_allclose(y, -0.4 * (state @ key), rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            updated @ orthogonal, 0.8 * (state @ orthogonal), rtol=0, atol=1e-14
        )

    def test_delta_uses_supplied_query_and_key_without_normalizing(self):
        state = np.eye(2)
        output, updated = delta_step(state, [2, 0], [2, 0], [4, 0], 1, 0.5)
        np.testing.assert_array_equal(updated, [[3, 0], [0, 1]])
        np.testing.assert_array_equal(output, [6, 0])
        np.testing.assert_array_equal(state, np.eye(2))

    def test_delta_validates_gates_shapes_and_real_values(self):
        state, key = np.eye(2), [1, 0]
        with self.assertRaisesRegex(ValueError, "state shape"):
            delta_step(state, [1], key, key, 1, 1)
        with self.assertRaisesRegex(ValueError, "alpha"):
            delta_step(state, key, key, key, 1.1, 1)
        with self.assertRaisesRegex(ValueError, "beta"):
            delta_step(state, key, key, key, 1, 2.1)
        with self.assertRaisesRegex(ValueError, "real numbers"):
            delta_step(state, key, key, np.array([1j, 1]), 1, 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            delta_step(state, key, key, [1, np.nan], 1, 1)


class StackTests(unittest.TestCase):
    def test_attention_append_matches_causal_dense_with_distinct_widths(self):
        rng = np.random.default_rng(15)
        q, k = rng.normal(size=(2, 5, 3))
        v = rng.normal(size=(5, 2))
        cache = (np.empty((0, 3)), np.empty((0, 2)))
        incremental = []
        for position in range(len(q)):
            output, cache = attention_token(q[position], k[position], v[position], cache)
            incremental.append(output)
        mask = np.tril(np.ones((len(q), len(k)), dtype=bool))
        dense, _ = scaled_dot_product_attention(q, k, v, mask)
        np.testing.assert_allclose(incremental, dense, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(cache[0], k)
        np.testing.assert_array_equal(cache[1], v)

    def test_hybrid_retains_separate_stores_and_caches_layer_input(self):
        tokens = toy_tokens()
        stores = initial_stores(HYBRID)
        self.assertFalse(np.shares_memory(stores[0], stores[2]))
        for token in tokens[:3]:
            _, stores = stack_step(token, HYBRID, stores)
        snapshots = (stores[0].copy(), tuple(x.copy() for x in stores[1]), stores[2].copy())
        trace, next_stores = stack_step(tokens[3], HYBRID, stores)
        self.assertEqual(history_counts(stores), (4, 12, 4))
        self.assertEqual(history_counts(next_stores), (4, 16, 4))
        for layer in (0, 2):
            np.testing.assert_array_equal(stores[layer], snapshots[layer])
            self.assertFalse(np.shares_memory(stores[layer], next_stores[layer]))
        for original, snapshot in zip(stores[1], snapshots[1]):
            np.testing.assert_array_equal(original, snapshot)
        self.assertFalse(np.shares_memory(next_stores[0], next_stores[2]))
        np.testing.assert_array_equal(next_stores[1][0][-1], trace[1])
        np.testing.assert_array_equal(next_stores[1][1][-1], trace[1][::-1])
        np.testing.assert_array_equal(trace[0], tokens[3])

    def test_matched_three_layer_history_counts(self):
        tokens = toy_tokens()
        for schedule, expected in (
            (("delta",) * 3, [12] * 9),
            (HYBRID, [4 * n + 8 for n in range(9)]),
            (("attention",) * 3, [12 * n for n in range(9)]),
        ):
            with self.subTest(schedule=schedule):
                report = schedule_example(tokens, schedule)
                self.assertEqual(report["total_history_by_prefix"], expected)
        self.assertIsNone(schedule_example(tokens[:0], HYBRID)["last_output"])

    def test_all_attention_layers_match_full_causal_reference(self):
        tokens = toy_tokens()
        schedule = ("attention",) * 3
        stores = initial_stores(schedule)
        streaming = []
        for token in tokens:
            trace, stores = stack_step(token, schedule, stores)
            streaming.append(trace[-1])
        h = tokens
        mask = np.tril(np.ones((len(tokens), len(tokens)), dtype=bool))
        for _ in schedule:
            h, _ = scaled_dot_product_attention(h, h, h[:, ::-1], mask)
        np.testing.assert_allclose(streaming, h, rtol=0, atol=1e-14)

    def test_zero_token_and_invalid_history_shapes(self):
        stores = initial_stores(HYBRID)
        with np.errstate(all="raise"):
            trace, grown = stack_step([0, 0], HYBRID, stores)
        np.testing.assert_array_equal(trace, np.zeros((4, 2)))
        self.assertEqual(history_counts(grown), (4, 4, 4))
        with self.assertRaisesRegex(ValueError, "one store per layer"):
            stack_step([1, 1], HYBRID, stores[:2])
        with self.assertRaisesRegex(ValueError, "match length"):
            attention_token([1, 0], [1, 0], [1, 0], (np.zeros((2, 2)), np.zeros((1, 2))))


if __name__ == "__main__":
    unittest.main()
