"""Check the MLA attention algebra against explicit per-head calculations."""

import json
import unittest

import numpy as np

from examples.ch10_kv.latent import (
    expanded_attention,
    latent_attention,
    latent_demo,
    latent_inputs,
)


class LatentAttentionTests(unittest.TestCase):
    def test_two_head_scores_weights_and_values(self):
        # The first head's two content products are 8 and 4. Its rotary
        # products are 0 and 1. For head two they are 1 and -2, then 1 and 0.
        scores = np.array([[4.0, 2.5], [1.0, -1.0]])
        exp_scores = np.exp(scores)
        expected_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        expected_output = np.array([
            expected_weights[0] @ [[1, 4], [2, 0]],
            expected_weights[1] @ [[4, 1], [0, 2]],
        ])
        for attention in (expanded_attention, latent_attention):
            with self.subTest(attention=attention.__name__):
                output, weights = attention(*latent_inputs())
                np.testing.assert_allclose(weights, expected_weights, rtol=0, atol=1e-14)
                np.testing.assert_allclose(output, expected_output, rtol=0, atol=1e-14)

    def test_wider_latent_does_not_change_score_scale(self):
        q_content, q_rotary, latent, k_rotary, w_uk, w_uv = latent_inputs()
        expected_output, expected_weights = expanded_attention(
            q_content, q_rotary, latent, k_rotary, w_uk, w_uv
        )
        # Add a latent feature ignored by all up-projections. The expanded
        # keys and values are unchanged, as is their score denominator.
        wider_latent = np.column_stack((latent, [3, -5]))
        wider_uk = np.pad(w_uk, ((0, 0), (0, 1), (0, 0)))
        wider_uv = np.pad(w_uv, ((0, 0), (0, 1), (0, 0)))
        output, weights = latent_attention(
            q_content, q_rotary, wider_latent, k_rotary, wider_uk, wider_uv
        )
        np.testing.assert_allclose(weights, expected_weights, rtol=0, atol=1e-14)
        np.testing.assert_allclose(output, expected_output, rtol=0, atol=1e-14)

    def test_rejects_mismatched_head_count(self):
        q_content, q_rotary, latent, k_rotary, w_uk, w_uv = latent_inputs()
        for attention in (expanded_attention, latent_attention):
            with self.subTest(attention=attention.__name__):
                with self.assertRaisesRegex(ValueError, "key projections"):
                    attention(q_content, q_rotary, latent, k_rotary, w_uk[:1], w_uv)

    def test_rejects_complex_input_before_float_conversion(self):
        q_content, q_rotary, latent, k_rotary, w_uk, w_uv = latent_inputs()
        q_content = q_content + 20j
        for attention in (expanded_attention, latent_attention):
            with self.subTest(attention=attention.__name__):
                with self.assertRaisesRegex(ValueError, "real numbers"):
                    attention(q_content, q_rotary, latent, k_rotary, w_uk, w_uv)

    def test_rejects_overflowing_value_projection(self):
        query = np.zeros((1, 1))
        rotary_key = np.zeros((1, 1))
        latent = np.array([[1e200]])
        w_uk = np.zeros((1, 1, 1))
        w_uv = np.array([[[1e200]]])
        for attention in (expanded_attention, latent_attention):
            with self.subTest(attention=attention.__name__), np.errstate(over="ignore"):
                with self.assertRaisesRegex(ValueError, "finite"):
                    attention(query, query, latent, rotary_key, w_uk, w_uv)

    def test_demo_reports_hand_checks_and_representation_sizes(self):
        report = latent_demo()
        self.assertEqual(report["cached_scalars"], 8)
        self.assertEqual(report["expanded_scalars_with_shared_rotary"], 20)
        self.assertLessEqual(report["max_output_difference"], 1e-14)
        self.assertLessEqual(report["max_weight_difference"], 1e-14)
        self.assertEqual(report["hand_checks"], {
            "expanded_content_dot": 8.0,
            "absorbed_content_dot": 8.0,
            "expanded_weighted_value": [1.75, 1.0],
            "absorbed_weighted_value": [1.75, 1.0],
        })
        json.dumps(report, allow_nan=False)
