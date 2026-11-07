"""Numerical references and information-flow checks for Chapter 8."""

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from torch.nn import functional as F

    from examples.ch05_transformer.model import build_masks
    from examples.ch08_modern.model import (
        ModernBlock, ModernDecoder, RMSNorm, RotarySelfAttention, SwiGLU,
        rms_norm, rotate_pairs,
    )
    from examples.ch08_modern.run_examples import (
        one_update, pair_score_example, prepared_batch, rotation_example, swiglu_example,
    )


@unittest.skipIf(torch is None, "Chapter 8 requires PyTorch")
class ModernDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(81)

    def test_rms_norm_matches_hand_result_and_pytorch_on_feature_axis(self):
        x = torch.tensor([3.0, 4.0], dtype=torch.float64)
        torch.testing.assert_close(rms_norm(x, torch.ones_like(x), eps=0.0),
                                   x / math.sqrt(12.5))
        x = torch.randn(2, 3, 4, dtype=torch.float64)
        layer = RMSNorm(4).double()
        with torch.no_grad():
            layer.weight.copy_(x.new_tensor([0.4, -0.8, 1.2, 1.7]))
        result = layer(x)
        self.assertEqual(result.dtype, torch.float64)
        torch.testing.assert_close(
            result, F.rms_norm(x, (4,), layer.weight, eps=layer.eps),
            atol=1e-12, rtol=1e-12,
        )

    def test_rms_norm_promotes_half_statistics_and_handles_zero_inputs(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[300.0, -400.0], [0.0, 0.0]], dtype=dtype)
                weight = torch.tensor([0.5, 1.5], dtype=dtype)
                actual = rms_norm(x, weight)
                expected = F.rms_norm(x.float(), (2,), weight.float(), eps=1e-6)
                self.assertEqual(actual.dtype, dtype)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected.to(dtype), atol=0, rtol=0)

    def test_swiglu_matches_hand_product_and_width_budget(self):
        example = swiglu_example()
        negative = -2.0 / (1.0 + math.exp(-2.0))
        self.assertEqual(example["gate_projection"], [0.0, 2.0])
        self.assertEqual(example["up_projection"], [3.0, -1.0])
        for actual, expected in zip(example["output"], [0.0, negative, 0.0, -negative]):
            self.assertAlmostEqual(actual, expected, places=12)
        layer = SwiGLU(6, 16)
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 2 * 6 * 24)

    def test_rope_uses_adjacent_pairs_and_preserves_each_head_norm(self):
        x = torch.randn(2, 3, 2, 4, dtype=torch.float64)
        positions = torch.tensor([0, 3])
        expected = x.clone()
        for pair, omega in enumerate((1.0, 0.01)):
            angle = 3 * omega
            matrix = x.new_tensor([[math.cos(angle), -math.sin(angle)],
                                   [math.sin(angle), math.cos(angle)]])
            features = slice(2 * pair, 2 * pair + 2)
            expected[:, :, 1, features] = x[:, :, 1, features] @ matrix.T
        actual = rotate_pairs(x, positions)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual.norm(dim=-1), x.norm(dim=-1))

    def test_rope_relative_score_has_the_right_sign_after_shared_shift(self):
        q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
        k = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float64)
        for m, n in ((1, 0), (6, 5)):
            q_rot = rotate_pairs(q, torch.tensor([m]))
            k_rot = rotate_pairs(k, torch.tensor([n]))
            self.assertAlmostEqual((q_rot * k_rot).sum().item(), math.sin(1), places=12)
        for score in rotation_example()["pairwise_dot_products"]:
            self.assertAlmostEqual(score, 0.5, places=12)

    def test_rope_rejects_odd_width_and_wrong_positions(self):
        with self.assertRaisesRegex(ValueError, "even head_width"):
            rotate_pairs(torch.ones(1, 1, 2, 3), torch.arange(2))
        x = torch.ones(1, 1, 2, 4)
        with self.assertRaisesRegex(ValueError, "one index per token"):
            rotate_pairs(x, torch.arange(3))
        with self.assertRaisesRegex(TypeError, "integers"):
            rotate_pairs(x, torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "even head width"):
            RotarySelfAttention(6, 2)

    def test_two_frequency_head_sums_pair_scores_before_scaling(self):
        result = pair_score_example()
        expected = [math.cos(2), math.cos(0.02)]
        for actual, reference in zip(result["pair_contributions"], expected):
            self.assertAlmostEqual(actual, reference, places=12)
        self.assertAlmostEqual(result["unscaled_score"], sum(expected), places=12)
        self.assertAlmostEqual(result["scaled_logit"], sum(expected) / 2, places=12)

    def test_shared_shift_is_different_from_prepending_content(self):
        model = ModernDecoder().double().eval()
        original = torch.tensor([[7, 8, 9, 10]])
        prepended = torch.tensor([[16, 7, 8, 9, 10]])
        with torch.no_grad():
            baseline = model(original)
            shifted = model(original, start_pos=7)
            added_context = model(prepended)[:, 1:]
        torch.testing.assert_close(baseline, shifted, atol=1e-12, rtol=0)
        self.assertGreater((added_context - baseline).abs().max().item(), 1e-6)

    def test_attention_matches_pytorch_with_unrotated_values_and_boolean_mask(self):
        layer = RotarySelfAttention(8, 2).double()
        x = torch.randn(2, 3, 8, dtype=torch.float64)
        valid = torch.tensor([[True, True, True], [True, True, False]])
        _, allowed = build_masks(valid, valid)
        positions = torch.tensor([2, 3, 4])
        actual, weights = layer(x, allowed, positions)
        q = rotate_pairs(layer.split_heads(layer.q_proj(x)), positions)
        k = rotate_pairs(layer.split_heads(layer.k_proj(x)), positions)
        v = layer.split_heads(layer.v_proj(x))
        expected_heads = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        expected = layer.out_proj(layer.merge_heads(expected_heads))
        self.assertEqual(weights.shape, (2, 2, 3, 3))
        blocked = weights.masked_select(~allowed.expand_as(weights))
        self.assertTrue((blocked == 0).all())
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_zero_updates_leave_both_residual_routes_as_identity(self):
        block = ModernBlock(8, 2, 12).double()
        x = torch.randn(1, 3, 8, dtype=torch.float64, requires_grad=True)
        valid = torch.ones(1, 3, dtype=torch.bool)
        _, allowed = build_masks(valid, valid)
        with torch.no_grad():
            block.attention.out_proj.weight.zero_()
            block.feed_forward.down_proj.weight.zero_()
        result = block(x, allowed, torch.arange(3))
        torch.testing.assert_close(result, x, atol=0, rtol=0)
        (gradient,) = torch.autograd.grad(result.sum(), x)
        torch.testing.assert_close(gradient, torch.ones_like(x), atol=0, rtol=0)

    def test_decoder_shapes_and_parameter_budget(self):
        model = ModernDecoder()
        ids, labels = prepared_batch()
        self.assertEqual(model(ids).shape, (2, 3, 17))
        self.assertEqual(labels.shape, (2, 3))
        self.assertEqual(sum(p.numel() for p in model.blocks[0].parameters()), 2592)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5744)
        self.assertIsNot(model.embedding.weight, model.vocabulary_projection.weight)

    def test_later_input_changes_only_available_predictions(self):
        model = ModernDecoder().double().eval()
        first = torch.tensor([[7, 8, 9, 10]])
        second = torch.tensor([[7, 8, 9, 16]])
        with torch.no_grad():
            a, b = model(first), model(second)
        torch.testing.assert_close(a[:, :3], b[:, :3], atol=1e-12, rtol=0)
        self.assertGreater((a[:, 3] - b[:, 3]).abs().max().item(), 1e-6)

    def test_appended_padding_and_shared_position_shift_keep_real_logits(self):
        model = ModernDecoder().double().eval()
        ids, _ = prepared_batch()
        padded = F.pad(ids, (0, 2), value=model.pad_id)
        valid = ids != model.pad_id
        with torch.no_grad():
            expected = model(ids)[valid]
            appended = model(padded)[:, :ids.shape[1]][valid]
            shifted = model(ids, start_pos=13)[valid]
        torch.testing.assert_close(appended, expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(shifted, expected, atol=1e-12, rtol=0)

    def test_decoder_rejects_an_empty_prefix_and_internal_padding(self):
        model = ModernDecoder()
        for ids in ([[0, 0]], [[7, 0, 9]]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "right-padded"):
                model(torch.tensor(ids))

    def test_one_update_reduces_loss_with_finite_gradients(self):
        model = ModernDecoder().double()
        ids, labels = prepared_batch()
        update = one_update(model, ids, labels)
        self.assertTrue(update["finite_gradients"])
        self.assertTrue(math.isfinite(update["loss_before"]))
        self.assertTrue(math.isfinite(update["loss_after"]))
        self.assertLess(update["loss_after"], update["loss_before"])

    def test_one_update_rejects_nonfinite_gradient_before_changing_parameters(self):
        model = ModernDecoder().double()
        ids, labels = prepared_batch()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        hook = model.vocabulary_projection.weight.register_hook(
            lambda grad: torch.full_like(grad, float("nan"))
        )
        self.addCleanup(hook.remove)

        with self.assertRaisesRegex(FloatingPointError, "finite gradient"):
            one_update(model, ids, labels)

        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
