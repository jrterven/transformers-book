"""Check Chapter 10's masks, compact cache, and grouped attention."""

import unittest

import numpy as np

from examples.ch04_attention.attention import scaled_dot_product_attention
from examples.ch10_kv.model import (
    attention_step,
    cache_bytes,
    causal_mask,
    grouped_attention,
)
from examples.ch10_kv.run_examples import hand_example, projected_example, run_chunks


class AttentionTests(unittest.TestCase):
    def test_hand_example(self):
        record = hand_example()
        np.testing.assert_allclose(record["scores"], [1 / np.sqrt(2), 0, 1 / np.sqrt(2)])
        np.testing.assert_allclose(record["weights"], [
            0.4011120926797859, 0.1977758146404282, 0.4011120926797859,
        ], rtol=0, atol=1e-15)
        np.testing.assert_allclose(record["output"], [1.6044483707191436, 1])

    def test_prefill_and_absolute_append_masks(self):
        np.testing.assert_array_equal(causal_mask(0, 3, 3), [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ])
        np.testing.assert_array_equal(causal_mask(3, 3, 6), [
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ])
        np.testing.assert_array_equal(causal_mask(5, 1, 6), np.ones((1, 6), dtype=bool))

    def test_grouped_matches_explicitly_repeated_keys_and_values(self):
        rng = np.random.default_rng(12)
        q = rng.normal(size=(2, 4, 3, 2))
        allowed = causal_mask(2, 3, 5)
        for kv_heads in (4, 2, 1):
            with self.subTest(kv_heads=kv_heads):
                k = rng.normal(size=(2, kv_heads, 5, 2))
                v = rng.normal(size=(2, kv_heads, 5, 3))
                output, weights = grouped_attention(q, k, v, allowed)
                repeated_k = np.repeat(k, 4 // kv_heads, axis=1)
                repeated_v = np.repeat(v, 4 // kv_heads, axis=1)
                reference_output, reference_weights = scaled_dot_product_attention(
                    q, repeated_k, repeated_v, allowed
                )
                np.testing.assert_allclose(output, reference_output, rtol=0, atol=1e-14)
                np.testing.assert_allclose(weights, reference_weights, rtol=0, atol=1e-14)

    def test_invalid_query_head_count(self):
        q = np.zeros((1, 3, 2, 2))
        k = v = np.zeros((1, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "divisible"):
            grouped_attention(q, k, v)


class CacheTests(unittest.TestCase):
    def test_full_causal_matches_both_incremental_plans(self):
        q, keys, values = projected_example()
        for kv_heads, expected_bytes in ((4, 768), (2, 384), (1, 192)):
            k, v = keys[:, :kv_heads], values[:, :kv_heads]
            full, _ = grouped_attention(q, k, v, causal_mask(0, 6, 6))
            for chunks in ((2, 1, 1, 1, 1), (2, 1, 3)):
                with self.subTest(kv_heads=kv_heads, chunks=chunks):
                    output, (cached_k, cached_v) = run_chunks(q, k, v, chunks)
                    np.testing.assert_allclose(output, full, rtol=0, atol=1e-12)
                    self.assertEqual(cached_k.shape, (1, kv_heads, 6, 2))
                    self.assertEqual(cached_v.shape, (1, kv_heads, 6, 2))
                    self.assertEqual(cached_k.nbytes + cached_v.nbytes, expected_bytes)
                    np.testing.assert_array_equal(cached_k, k)
                    np.testing.assert_array_equal(cached_v, v)

    def test_future_perturbation_in_appended_chunk(self):
        q, k, v = projected_example()
        k, v = k[:, :2], v[:, :2]
        _, _, cache = attention_step(q[:, :, :3], k[:, :, :3], v[:, :, :3])
        tail = q[:, :, 3:], k[:, :, 3:], v[:, :, 3:]
        output, weights, _ = attention_step(*tail, cache=cache)
        changed_v = tail[2].copy()
        changed_v[:, :, -1] += 100
        changed, _, _ = attention_step(tail[0], tail[1], changed_v, cache=cache)
        np.testing.assert_array_equal(weights[..., 0, 4:], 0)
        np.testing.assert_array_equal(weights[..., 1, 5:], 0)
        np.testing.assert_allclose(output[:, :, :2], changed[:, :, :2], rtol=0, atol=1e-12)
        self.assertGreater(np.max(np.abs(output[:, :, -1] - changed[:, :, -1])), 1)

    def test_prefill_owns_arrays_and_append_keeps_previous_snapshot(self):
        q, k, v = projected_example()
        k, v = k[:, :1].astype(np.float32), v[:, :1].astype(np.float32)
        _, _, cache = attention_step(q[:, :, :2], k[:, :, :2], v[:, :, :2])
        _, _, grown = attention_step(q[:, :, 2:3], k[:, :, 2:3], v[:, :, 2:3], cache)
        for old, new, source in zip(cache, grown, (k, v)):
            np.testing.assert_array_equal(old, source[:, :, :2])
            np.testing.assert_array_equal(new, source[:, :, :3])
            self.assertFalse(np.shares_memory(old, source))
            self.assertFalse(np.shares_memory(old, new))
            self.assertEqual(new.dtype, np.float32)

    def test_mismatched_chunk_lengths_and_cache_dtype(self):
        q, k, v = projected_example()
        with self.assertRaisesRegex(ValueError, "one new key"):
            attention_step(q[:, :, :1], k[:, :, :2], v[:, :, :2])
        _, _, cache = attention_step(q[:, :, :2], k[:, :, :2], v[:, :, :2])
        with self.assertRaisesRegex(ValueError, "dtype"):
            attention_step(q[:, :, 2:3], k[:, :, 2:3].astype(np.float32),
                           v[:, :, 2:3], cache)

    def test_payload_bytes_including_unequal_widths(self):
        self.assertEqual(cache_bytes(2, 3, 5, 4, 6, 7, 2), 3120)
        self.assertEqual(cache_bytes(1, 1, 0, 1, 2, 2, 8), 0)
        for heads, expected_mib in ((16, 192), (4, 48), (1, 12)):
            self.assertEqual(cache_bytes(1, 24, 2048, heads, 64, 64, 2),
                             expected_mib * 2**20)


if __name__ == "__main__":
    unittest.main()
