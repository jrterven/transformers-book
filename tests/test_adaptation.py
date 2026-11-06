"""Check Chapter 16's updates, label alignment, and preference loss."""

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from torch import nn
    from torch.nn import functional as F

    from examples.ch16_adaptation.model import (
        LoRALinear,
        TinyClassifier,
        completion_logprob,
        dpo_loss,
        response_labels,
        response_loss,
        train_classifier,
    )
    from examples.ch16_adaptation.run_examples import (
        classification_data,
        classifier_example,
        lora_example,
        parameter_count,
        response_example_inputs,
    )


@unittest.skipIf(torch is None, "Chapter 16 requires PyTorch")
class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_same_classifier_initialization_and_update_boundaries(self):
        report = classifier_example()
        frozen, full = report["frozen_encoder"], report["full_fine_tuning"]
        self.assertEqual(frozen["held_out_before"], full["held_out_before"])
        self.assertEqual(frozen["trainable_parameters"], 8)
        self.assertEqual(full["trainable_parameters"], 17)
        self.assertEqual(frozen["max_encoder_change"], 0)
        self.assertGreater(full["max_encoder_change"], 0)
        for outcome in (frozen, full):
            self.assertEqual(outcome["total_parameters"], 17)
            self.assertEqual(outcome["held_out_after"]["accuracy"], 1)
            self.assertLess(outcome["held_out_after"]["loss"],
                            outcome["held_out_before"]["loss"])

    def test_freezing_encoder_removes_stale_gradients(self):
        train, labels, _, _ = classification_data()
        model = TinyClassifier()
        train_classifier(model, train, labels, freeze_encoder=False, steps=2)
        self.assertTrue(all(p.grad is not None for p in model.encoder.parameters()))
        before = [p.detach().clone() for p in model.encoder.parameters()]
        train_classifier(model, train, labels, freeze_encoder=True, steps=2)
        self.assertFalse(model.encoder.training)
        for p, old in zip(model.encoder.parameters(), before):
            self.assertIsNone(p.grad)
            torch.testing.assert_close(p, old, rtol=0, atol=0)

    def test_response_mask_shifts_together_with_ids_and_ignores_padding(self):
        ids, mask, logits = response_example_inputs()
        mask[:, -1] = True
        labels = response_labels(ids, mask)
        torch.testing.assert_close(labels, torch.tensor([[-100, -100, -100, 6, 2, -100]]))
        self.assertAlmostEqual(float(completion_logprob(logits, labels)[0]), -math.log(6))
        self.assertAlmostEqual(float(response_loss(logits, labels)), math.log(6) / 2)

    def test_ignored_targets_have_zero_direct_logit_gradients(self):
        ids, mask, logits = response_example_inputs()
        labels = response_labels(ids, mask)
        logits[:, [0, 1, 2, 5, 6], 0] = 10_000
        logits.requires_grad_()
        score = completion_logprob(logits, labels).sum()
        score.backward()
        self.assertAlmostEqual(float(score.detach()), -math.log(6))
        self.assertEqual(int(torch.count_nonzero(logits.grad[:, [0, 1, 2, 5, 6]])), 0)
        self.assertGreater(float(logits.grad[0, 3, 6]), 0)
        self.assertGreater(float(logits.grad[0, 4, 2]), 0)

    def test_each_response_gets_a_sum_and_half_precision_promotes(self):
        ids = torch.tensor([[1, 3, 2, 0], [1, 4, 4, 2]])
        mask = torch.tensor([[False, False, True, False], [False, False, True, True]])
        labels = response_labels(ids, mask)
        logits = torch.zeros((2, 4, 5), dtype=torch.float16)
        expected = -math.log(5) * torch.tensor([1.0, 2.0], dtype=torch.float32)
        torch.testing.assert_close(completion_logprob(logits, labels), expected)
        self.assertAlmostEqual(float(response_loss(logits, labels)), math.log(5), places=6)

    def test_response_validation_rejects_ambiguous_or_invalid_alignment(self):
        ids, mask, logits = response_example_inputs()
        with self.assertRaisesRegex(ValueError, "boolean"):
            response_labels(ids, mask.long())
        bad = mask.clone()
        bad[:, 0] = True
        with self.assertRaisesRegex(ValueError, "earlier logit"):
            response_labels(ids, bad)
        with self.assertRaisesRegex(ValueError, "right-padded"):
            response_labels(torch.tensor([[1, 0, 2]]), torch.tensor([[False, False, True]]))
        with self.assertRaisesRegex(ValueError, "response target"):
            response_labels(ids, torch.zeros_like(mask))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            completion_logprob(logits[:, :, :5], response_labels(ids, mask))

    def test_worked_lora_and_checkpoint(self):
        report = lora_example()
        self.assertTrue(report["initial_b_is_zero"])
        self.assertEqual(report["initial_output"], [5, -1])
        self.assertEqual(report["scaled_update"], [2, 4])
        for name in ("adapted_output", "merged_output", "restored_output"):
            self.assertEqual(report[name], [7, 3])
        self.assertEqual(report["rank_of_update"], 1)
        self.assertEqual(report["larger_base_parameters"], 54)
        self.assertEqual(report["larger_adapter_parameters"], 28)

    def test_lora_training_changes_adapter_only_and_preserves_bias(self):
        base = nn.Linear(3, 2, dtype=torch.float64)
        layer = LoRALinear(base, rank=1, alpha=2)
        x = torch.tensor([[1, 2, 0], [3, 1, 2]], dtype=torch.float64)
        with torch.no_grad():
            layer.a.copy_(torch.tensor([[1, -1, 0.5]], dtype=torch.float64))
        original = [p.detach().clone() for p in base.parameters()]
        target = base(x) + 1
        optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.01)
        self.assertEqual(parameter_count(layer, trainable_only=True), 5)
        loss = F.mse_loss(layer(x), target)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(layer.a.grad)), 0)
        self.assertGreater(int(torch.count_nonzero(layer.b.grad)), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        F.mse_loss(layer(x), target).backward()
        self.assertGreater(int(torch.count_nonzero(layer.a.grad)), 0)
        for p, old in zip(base.parameters(), original):
            self.assertIsNone(p.grad)
            torch.testing.assert_close(p, old, atol=0, rtol=0)
        torch.testing.assert_close(layer(x), F.linear(x, layer.merged_weight(), base.bias))

    def test_dpo_value_and_gradients_keep_reference_fixed(self):
        policy = torch.tensor([-3.0, -4.0], dtype=torch.float64, requires_grad=True)
        reference = torch.tensor([-4.0, -4.5], dtype=torch.float64, requires_grad=True)
        loss = dpo_loss(*policy, *reference, beta=2)
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), math.log1p(math.exp(-1)))
        expected = 2 / (1 + math.exp(1))
        torch.testing.assert_close(policy.grad, policy.new_tensor([-expected, expected]))
        self.assertIsNone(reference.grad)

    def test_dpo_stays_finite_for_large_half_precision_margins(self):
        policy_chosen = torch.tensor([0, -60_000], dtype=torch.float16)
        policy_rejected = torch.tensor([-60_000, 0], dtype=torch.float16)
        reference = torch.zeros_like(policy_chosen)
        loss = dpo_loss(policy_chosen, policy_rejected, reference, reference, beta=2)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(float(loss), 60_000)
        with self.assertRaisesRegex(ValueError, "positive"):
            dpo_loss(policy_chosen, policy_rejected, reference, reference, beta=0)


if __name__ == "__main__":
    unittest.main()
