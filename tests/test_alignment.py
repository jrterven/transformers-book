"""Numerical and information-flow checks for learned additive attention."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from examples.ch03_alignment.model import additive_scores, weighted_read

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from examples.ch02_recurrence.data import BOS, EOS, PAD, collate
    from examples.ch02_recurrence.model import token_loss
    from examples.ch03_alignment.model import AdditiveAttention, AttentiveSeq2Seq, greedy_decode
    from examples.ch03_alignment.run_examples import alignment_example, evaluate


class AlignmentArithmeticTests(unittest.TestCase):
    def test_given_scores_reproduce_the_context_by_hand(self):
        annotations = np.array([[1., 0.], [0., 1.], [1., 1.]])
        weights, context = weighted_read(np.log([.05, .10, .85]), annotations)
        np.testing.assert_allclose(weights, [.05, .10, .85], atol=1e-15)
        np.testing.assert_allclose(context, [.9, .95], atol=1e-15)

    def test_additive_scores_match_independent_coordinate_sums(self):
        rng = np.random.default_rng(21)
        annotations, query = rng.normal(size=(4, 3)), rng.normal(size=2)
        wa, wq = rng.normal(size=(3, 5)), rng.normal(size=(2, 5))
        bias, direction = rng.normal(size=5), rng.normal(size=5)
        actual = additive_scores(query, annotations, wq, wa, direction, bias)
        expected = []
        for row in annotations:
            score = 0.
            for k in range(5):
                z = (sum(row[d] * wa[d, k] for d in range(3)) +
                     sum(query[d] * wq[d, k] for d in range(2)) + bias[k])
                score += direction[k] * math.tanh(z)
            expected.append(score)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)

    def test_column_direction_preserves_one_score_per_source(self):
        annotations = np.array([[1., 0.], [0., 1.], [1., 1.]])
        query = np.array([.2, -.1])
        direction = np.array([1., -1.])
        flat = additive_scores(query, annotations, np.eye(2), np.eye(2), direction)
        column = additive_scores(query, annotations, np.eye(2), np.eye(2),
                                 direction[:, None])
        self.assertEqual(column.shape, (3,))
        np.testing.assert_array_equal(column, flat)
        weights, context = weighted_read(column, annotations)
        np.testing.assert_allclose(weights, [.596567504375604, .139623183989087,
                                             .263809311635308], atol=1e-14, rtol=0)
        np.testing.assert_allclose(context, [.860376816010913, .403432495624396],
                                   atol=1e-14, rtol=0)
        with self.assertRaisesRegex(ValueError, "direction"):
            additive_scores(query, annotations, np.eye(2), np.eye(2), np.eye(2))

    def test_weighted_read_broadcasts_batches_but_never_source_positions(self):
        annotations = np.array([[1., 0.], [0., 1.], [1., 1.]])
        scores = np.array([np.log([.05, .10, .85]), [0., 0., 0.]])
        weights, context = weighted_read(scores, annotations)
        self.assertEqual(weights.shape, (2, 3))
        np.testing.assert_allclose(context, [[.9, .95], [2 / 3, 2 / 3]],
                                   atol=1e-15, rtol=0)
        # One common score row can also read two independent annotation batches.
        annotation_batch = np.stack((annotations, annotations + 2))
        shared_weights, batch_context = weighted_read(scores[0], annotation_batch)
        self.assertEqual(shared_weights.shape, (3,))
        np.testing.assert_allclose(batch_context, [[.9, .95], [2.9, 2.95]],
                                   atol=1e-15, rtol=0)
        # A singleton source axis must not silently repeat its weight three times.
        for wrong_scores in ([0.], np.zeros((3, 1))):
            with self.assertRaisesRegex(ValueError, "one score"):
                weighted_read(wrong_scores, annotations)
        with self.assertRaisesRegex(ValueError, "batch axes"):
            weighted_read(np.zeros((2, 3)), np.zeros((4, 3, 2)))

    def test_numpy_mask_and_score_shift_and_all_invalid_rows(self):
        annotations = np.array([[1., 2.], [3., 4.], [100., 200.]])
        mask = np.array([True, True, False])
        weights, context = weighted_read([1000., 1001., 5000.], annotations, mask)
        expected = np.array([1, math.e, 0]) / (1 + math.e)
        np.testing.assert_allclose(weights, expected, atol=1e-15)
        np.testing.assert_allclose(context, expected @ annotations, atol=1e-15)
        shifted, _ = weighted_read([17., 18., 4017.], annotations, mask)
        np.testing.assert_allclose(weights, shifted, atol=1e-15)
        with self.assertRaises(ValueError):
            weighted_read([1., 2., 3.], annotations, np.zeros(3, dtype=bool))


@unittest.skipIf(torch is None, "Chapter 3 recurrent attention requires PyTorch")
class AttentiveSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(31)
        self.model = AttentiveSeq2Seq(embedding_width=4, hidden_width=5,
                                     alignment_width=3).double().eval()
        self.batch = collate([(3, 4, 5), (6, 3), (8, 5, 7, 3)])

    def test_torch_attention_matches_hand_parameter_numpy_example(self):
        attention = AdditiveAttention(2, 2).double()
        with torch.no_grad():
            attention.annotation.weight.copy_(torch.eye(2))
            attention.query.weight.copy_(torch.eye(2))
            attention.energy.weight.copy_(torch.tensor([[1., -1.]]))
            attention.bias.zero_()
        annotations = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]]).double()
        query = torch.tensor([[.2, -.1]], dtype=torch.float64)
        valid = torch.ones(1, 3, dtype=torch.bool)
        context, weights = attention(query, annotations, attention.annotation(annotations), valid)
        scores = np.array([math.tanh(1.2) - math.tanh(-.1),
                           math.tanh(.2) - math.tanh(.9),
                           math.tanh(1.2) - math.tanh(.9)])
        expected = np.exp(scores - scores.max())
        expected /= expected.sum()
        np.testing.assert_allclose(weights.detach(), expected[None], atol=2e-13, rtol=0)
        np.testing.assert_allclose(context.detach(), (expected @ annotations[0].numpy())[None],
                                   atol=2e-13, rtol=0)

    def test_attention_rows_normalize_only_over_valid_source_positions(self):
        logits, weights = self.model(self.batch.source, self.batch.lengths,
                                    self.batch.decoder_input)
        self.assertEqual(tuple(logits.shape), (3, 5, 9))
        self.assertEqual(tuple(weights.shape), (3, 5, 4))
        torch.testing.assert_close(weights.sum(-1), torch.ones(3, 5).double(),
                                   atol=2e-16, rtol=1e-14)
        for row, length in enumerate(self.batch.lengths.tolist()):
            self.assertEqual(torch.count_nonzero(weights[row, :, length:]).item(), 0)

    def test_all_invalid_source_is_rejected(self):
        attention = self.model.attention
        a = torch.zeros(1, 3, 5).double()
        with self.assertRaises(ValueError):
            attention(torch.zeros(1, 5).double(), a, attention.annotation(a),
                      torch.zeros(1, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.model.encode(torch.zeros(1, 3, dtype=torch.long), torch.tensor([3]))
        with self.assertRaises(ValueError):
            self.model.encode(torch.tensor([[3, 4]]), torch.tensor([0]))

    def test_source_padding_does_not_change_states_logits_or_real_weights(self):
        batch = self.batch
        memory, state = self.model.encode(batch.source, batch.lengths)
        original_logits, original_weights = self.model(batch.source, batch.lengths,
                                                       batch.decoder_input)
        changed = torch.nn.functional.pad(batch.source, (0, 3), value=8)
        for row, length in enumerate(batch.lengths.tolist()):
            changed[row, length:] = 8
        new_memory, new_state = self.model.encode(changed, batch.lengths)
        logits, weights = self.model(changed, batch.lengths, batch.decoder_input)
        for a, b in zip(state, new_state):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(memory[0], new_memory[0][:, :4], atol=0, rtol=0)
        torch.testing.assert_close(logits, original_logits, atol=1e-15, rtol=1e-13)
        torch.testing.assert_close(weights[:, :, :4], original_weights, atol=1e-15, rtol=1e-13)
        self.assertEqual(weights[:, :, 4:].count_nonzero().item(), 0)

    def test_teacher_forward_has_no_future_token_leak_and_has_attention_gradients(self):
        b = self.batch
        logits, weights = self.model(b.source, b.lengths, b.decoder_input)
        changed = b.decoder_input.clone()
        changed[:, 3:] = 7
        new_logits, new_weights = self.model(b.source, b.lengths, changed)
        torch.testing.assert_close(logits[:, :3], new_logits[:, :3], atol=0, rtol=0)
        torch.testing.assert_close(weights[:, :3], new_weights[:, :3], atol=0, rtol=0)
        token_loss(logits, b.target).backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for parameter in self.model.attention.parameters():
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_masked_source_annotations_receive_no_gradient(self):
        attention = self.model.attention
        annotations = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
        query = torch.randn(2, 5, dtype=torch.float64)
        valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
        context, _ = attention(query, annotations, attention.annotation(annotations), valid)
        context.square().sum().backward()
        torch.testing.assert_close(annotations.grad[~valid],
                                   torch.zeros_like(annotations.grad[~valid]), atol=0, rtol=0)
        self.assertGreater(annotations.grad[valid].abs().sum().item(), 0)

    def test_greedy_alignment_matches_its_supplied_prefix(self):
        b = self.batch
        tokens, attention = greedy_decode(self.model, b.source, b.lengths, 5)
        inputs = torch.cat((torch.full((3, 1), BOS), tokens[:, :-1]), dim=1)
        logits, reference_attention = self.model(b.source, b.lengths, inputs)
        for row in range(3):
            ids = tokens[row].tolist()
            count = ids.index(EOS) + 1 if EOS in ids else len(ids)
            torch.testing.assert_close(tokens[row, :count], logits[row, :count].argmax(-1))
            torch.testing.assert_close(attention[row, :count], reference_attention[row, :count],
                                       atol=1e-15, rtol=1e-13)

    def test_batched_greedy_matches_individual_sequences(self):
        b = self.batch
        tokens, weights = greedy_decode(self.model, b.source, b.lengths, 5)
        for row, length in enumerate(b.lengths.tolist()):
            individual, individual_weights = greedy_decode(
                self.model, b.source[row:row + 1, :length], torch.tensor([length]), 5)
            ids = individual[0].tolist()
            count = ids.index(EOS) + 1 if EOS in ids else len(ids)
            torch.testing.assert_close(tokens[row, :count], individual[0, :count])
            torch.testing.assert_close(weights[row, :count, :length],
                                       individual_weights[0, :count], atol=1e-15, rtol=1e-13)

    def test_eos_attention_is_kept_but_later_finished_rows_are_zero(self):
        b = self.batch
        first, second = torch.zeros(3, 9), torch.zeros(3, 9)
        first[0, EOS], first[1, 4], first[2, EOS] = 1, 1, 1
        second[:, EOS] = 1
        with patch.object(self.model.output, "forward", side_effect=[first, second]):
            tokens, weights = greedy_decode(self.model, b.source, b.lengths, 7)
        self.assertEqual(tokens.tolist(), [[EOS, PAD], [4, EOS], [EOS, PAD]])
        torch.testing.assert_close(weights[:, 0].sum(-1), torch.ones(3).double(),
                                   atol=1e-15, rtol=1e-13)
        self.assertEqual(weights[[0, 2], 1].count_nonzero().item(), 0)
        self.assertAlmostEqual(weights[1, 1].sum().item(), 1., places=14)
        with self.assertRaises(ValueError):
            greedy_decode(self.model, b.source, b.lengths, 0)

    def test_evaluation_token_weighting_and_eos_required_exact_match(self):
        sources = [(3,), (4, 5), (6, 7, 8), (3, 4, 5, 6)]
        batch = collate(sources)
        correct_probabilities = [.9, .7, .5, .1]
        probabilities = torch.full((4, 5, 9), 1 / 9, dtype=torch.float64)
        for row, probability in enumerate(correct_probabilities):
            for pos, target in enumerate(batch.target[row].tolist()):
                if target != PAD:
                    probabilities[row, pos].fill_((1 - probability) / 8)
                    probabilities[row, pos, target] = probability
        generated = torch.tensor([[3, EOS, PAD, PAD, PAD],
                                  [BOS, 5, 4, EOS, PAD],
                                  [PAD, 8, 7, 6, EOS], [6, 5, 4, 3, 3]])
        with patch.object(self.model, "forward", return_value=(probabilities.log(), None)), patch(
                "examples.ch03_alignment.run_examples.greedy_decode",
                return_value=(generated, None)) as decode:
            result = evaluate(self.model, sources, generation_cap=5)
        expected = -sum(n * math.log(p) for n, p in
                        zip([2, 3, 4, 5], correct_probabilities)) / 14
        self.assertAlmostEqual(result["token_nll"], expected, places=14)
        self.assertEqual(result["nonpadding_tokens"], 14)
        self.assertEqual(result["teacher_forced_correct_tokens"], 9)
        self.assertEqual(result["exact_matches"], 1)
        self.assertEqual(result["ended_in_eos"], 3)
        for length, probability in enumerate(correct_probabilities, 1):
            self.assertAlmostEqual(result["by_length"][str(length)]["token_nll"],
                                   -math.log(probability), places=14)
        self.assertEqual(decode.call_args.args[-1], 5)

    def test_failed_alignment_trace_reports_fallback_and_missing_eos(self):
        sources = [(3, 4, 5, 6, 7, 8), (3, 3, 3, 3, 3, 3)]
        with torch.no_grad():
            self.model.output.weight.zero_()
            self.model.output.bias.zero_()
            self.model.output.bias[PAD] = 10
        results = evaluate(self.model, sources, generation_cap=1)
        self.assertEqual(results["exact_matches"], 0)
        trace = alignment_example(self.model, sources, results["predictions"],
                                  generation_cap=1)
        self.assertEqual(trace["test_index"], 0)
        self.assertEqual(trace["generated"], [PAD])
        self.assertFalse(trace["ended_in_eos"])
        self.assertEqual(trace["selection"],
                         "first length-six test example; no correct length-six prediction")
        self.assertEqual(trace["mode"],
                         "greedy generation; generation cap reached without EOS")
        self.assertEqual(len(trace["attention"]), 1)
        self.assertEqual(len(trace["attention"][0]), 6)


if __name__ == "__main__":
    unittest.main()
