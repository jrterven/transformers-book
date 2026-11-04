"""Independent numerical, sequence-alignment, and decoding checks for Chapter 2."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from examples.ch02_recurrence.model import gated_cell_update, lstm_step, rnn_trace

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from examples.ch02_recurrence.data import BOS, EOS, PAD, collate, make_splits
    from examples.ch02_recurrence.model import (
        RecurrentSeq2Seq, greedy_decode, token_loss,
    )
    from examples.ch02_recurrence.run_examples import evaluate


class RecurrentArithmeticTests(unittest.TestCase):
    def test_scalar_recurrence_matches_hand_numbers_and_depends_on_order(self):
        actual = rnn_trace([0.2, 0.4, -0.1])
        np.testing.assert_allclose(actual, [0.197375320224904, 0.46108444483003735,
                                           0.1298057073036557], atol=5e-13, rtol=0)
        self.assertGreater(abs(actual[-1] - rnn_trace([-0.1, 0.4, 0.2])[-1]), 0.2)
        np.testing.assert_allclose(rnn_trace([0.2, 0.4], recurrent_weight=0),
                                   np.tanh([0.2, 0.4]), atol=1e-15)

    def test_lstm_gate_arithmetic_and_memory_limits(self):
        hidden, cell = gated_cell_update(0.5, 0.8, 0.5, -0.4, 0.6)
        self.assertAlmostEqual(cell, 0.2, places=15)
        self.assertAlmostEqual(hidden, 0.1184251921349424, places=14)
        _, remembered = gated_cell_update(0.5, 1.0, 0.0, -0.9, 1.0)
        _, replaced = gated_cell_update(0.5, 0.0, 1.0, -0.9, 1.0)
        self.assertEqual(remembered, 0.5)
        self.assertEqual(replaced, -0.9)


@unittest.skipIf(torch is None, "Chapter 2 sequence examples require PyTorch")
class RecurrentSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(31)
        self.model = RecurrentSeq2Seq(embedding_width=4, hidden_width=5).double().eval()
        self.batch = collate([(3, 4, 5), (6, 3), (8, 5, 7, 3)])

    def test_numpy_lstm_step_matches_independent_pytorch_cell(self):
        cell = torch.nn.LSTMCell(3, 4).double()
        rng = np.random.default_rng(31)
        x, h, c = rng.normal(size=(2, 3)), rng.normal(size=(2, 4)), rng.normal(size=(2, 4))
        arrays = [p.detach().numpy() for p in
                  (cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)]
        actual = lstm_step(x, h, c, arrays[0].T, arrays[1].T, arrays[2] + arrays[3])
        expected = cell(torch.from_numpy(x), (torch.from_numpy(h), torch.from_numpy(c)))
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e.detach().numpy(), atol=3e-16, rtol=1e-14)

    def test_splits_are_balanced_disjoint_unique_and_repeatable(self):
        splits = make_splits()
        self.assertEqual(splits, make_splits())
        self.assertNotEqual(splits, make_splits(seed=27))
        unique = list(map(set, splits))
        for rows, size in zip(splits, (192, 48, 48)):
            self.assertEqual(len(set(rows)), size)
            self.assertEqual({n: sum(len(r) == n for r in rows) for n in range(3, 7)},
                             {n: size // 4 for n in range(3, 7)})
        self.assertFalse(unique[0] & unique[1] or unique[0] & unique[2] or unique[1] & unique[2])

    def test_teacher_forcing_shift_and_eos_are_exact(self):
        self.assertEqual(self.batch.lengths.tolist(), [3, 2, 4])
        self.assertEqual(self.batch.decoder_input.tolist(),
                         [[BOS, 5, 4, 3, PAD], [BOS, 3, 6, PAD, PAD],
                          [BOS, 3, 7, 5, 8]])
        self.assertEqual(self.batch.target.tolist(),
                         [[5, 4, 3, EOS, PAD], [3, 6, EOS, PAD, PAD],
                          [3, 7, 5, 8, EOS]])
        with self.assertRaises(ValueError):
            collate([(PAD, 3)])

    def test_packed_source_ignores_padding_and_restores_batch_order(self):
        batched = self.model.encode(self.batch.source, self.batch.lengths)
        changed = self.batch.source.clone()
        for j, length in enumerate(self.batch.lengths.tolist()):
            changed[j, length:] = 8
        perturbed = self.model.encode(changed, self.batch.lengths)
        for original, after in zip(batched, perturbed):
            torch.testing.assert_close(original, after, atol=0, rtol=0)
        for j, length in enumerate(self.batch.lengths.tolist()):
            single = self.model.encode(self.batch.source[j:j + 1, :length],
                                       torch.tensor([length]))
            for whole, one in zip(batched, single):
                torch.testing.assert_close(whole[:, j:j + 1], one, atol=1e-15, rtol=1e-13)

    def test_teacher_forward_is_causal_and_gradients_are_finite(self):
        batch = self.batch
        logits = self.model(batch.source, batch.lengths, batch.decoder_input)
        changed = batch.decoder_input.clone()
        changed[:, 3:] = 7
        changed_logits = self.model(batch.source, batch.lengths, changed)
        torch.testing.assert_close(logits[:, :3], changed_logits[:, :3], atol=0, rtol=0)
        self.assertEqual(tuple(logits.shape), (3, 5, 9))
        token_loss(logits, batch.target).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters()))
        self.assertGreater(self.model.encoder.weight_ih_l0.grad.abs().sum().item(), 0)

    def test_first_greedy_prediction_matches_teacher_forced_first_step(self):
        batch = self.batch
        logits = self.model(batch.source, batch.lengths, batch.decoder_input)
        predicted = greedy_decode(self.model, batch.source, batch.lengths, 1)
        torch.testing.assert_close(predicted[:, 0], logits[:, 0].argmax(-1), atol=0, rtol=0)

    def test_padding_aware_loss_has_independent_value_and_gradient(self):
        logits = torch.zeros(2, 3, 9, dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[3, EOS, PAD], [4, 5, EOS]])
        loss = token_loss(logits, target)
        self.assertAlmostEqual(loss.item(), math.log(9), places=14)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0, 2], torch.zeros(9).double()))
        self.assertAlmostEqual(logits.grad[0, 0, 3].item(), (1 / 9 - 1) / 5, places=14)
        self.assertGreater(logits.grad[0, 0, PAD].item(), 0)
        with self.assertRaises(ValueError):
            token_loss(logits, torch.zeros_like(target))

    def test_greedy_stops_at_eos_and_pads_finished_rows(self):
        source, lengths = self.batch.source[:2], self.batch.lengths[:2]
        first, second = torch.zeros(2, 9), torch.zeros(2, 9)
        first[0, EOS], first[1, 4] = 1, 1
        second[0, 7], second[1, EOS] = 1, 1
        with patch.object(self.model.output, "forward", side_effect=[first, second]) as output:
            predicted = greedy_decode(self.model, source, lengths, 6)
        self.assertEqual(predicted.tolist(), [[EOS, PAD], [4, EOS]])
        self.assertEqual(output.call_count, 2)

    def test_greedy_cap_handles_a_model_that_never_predicts_eos(self):
        fixed = torch.zeros(3, 9)
        fixed[:, 4] = 1
        with patch.object(self.model.output, "forward", return_value=fixed) as output:
            predicted = greedy_decode(self.model, self.batch.source, self.batch.lengths, 4)
        self.assertEqual(predicted.tolist(), [[4] * 4] * 3)
        self.assertEqual(output.call_count, 4)
        with self.assertRaises(ValueError):
            greedy_decode(self.model, self.batch.source, self.batch.lengths, 0)

    def test_evaluation_weights_tokens_and_grades_complete_generated_sequences(self):
        sources = [(3,), (4, 5), (6, 7, 8), (3, 4, 5, 6)]
        references = [[3, EOS], [5, 4, EOS], [8, 7, 6, EOS],
                      [6, 5, 4, 3, EOS]]
        correct_probabilities = [0.9, 0.7, 0.5, 0.1]
        probabilities = torch.full((4, 5, 9), 1 / 9, dtype=torch.float64)
        for row, (reference, probability) in enumerate(
                zip(references, correct_probabilities)):
            for position, target in enumerate(reference):
                probabilities[row, position].fill_((1 - probability) / 8)
                probabilities[row, position, target] = probability
        logits = probabilities.log()
        # All four rows have five storage positions. Only 14 labels are real.
        generated = torch.tensor([
            [3, EOS, PAD, PAD, PAD],       # Correct; trailing storage is ignored.
            [BOS, 5, 4, EOS, PAD],         # An extra BOS is a generated error.
            [PAD, 8, 7, 6, EOS],           # An extra pre-EOS PAD is also an error.
            [6, 5, 4, 3, 3],              # Correct colors, but no EOS at the cap.
        ])
        with patch.object(self.model, "forward", return_value=logits), patch(
                "examples.ch02_recurrence.run_examples.greedy_decode",
                return_value=generated) as decode:
            actual = evaluate(self.model, sources, generation_cap=5)

        counts = [2, 3, 4, 5]
        expected_nll = -sum(n * math.log(p) for n, p in
                            zip(counts, correct_probabilities)) / 14
        sentence_mean = -sum(map(math.log, correct_probabilities)) / 4
        self.assertGreater(abs(expected_nll - sentence_mean), 0.1)
        self.assertAlmostEqual(actual["token_nll"], expected_nll, places=14)
        self.assertEqual(actual["nonpadding_tokens"], 14)
        # At p=.1 each other class has probability .1125, so all five
        # positions in the final row have an incorrect top-ranked token.
        self.assertEqual(actual["teacher_forced_correct_tokens"], 9)
        self.assertEqual(actual["sequences"], 4)
        self.assertEqual(actual["exact_matches"], 1)
        self.assertEqual(actual["ended_in_eos"], 3)
        self.assertEqual(actual["predictions"],
                         [[3, EOS], [BOS, 5, 4, EOS], [PAD, 8, 7, 6, EOS],
                          [6, 5, 4, 3, 3]])
        for length, probability in enumerate(correct_probabilities, 1):
            group = actual["by_length"][str(length)]
            self.assertEqual(group["sequences"], 1)
            self.assertAlmostEqual(group["token_nll"], -math.log(probability),
                                   places=14)
            self.assertEqual(group["exact_matches"], int(length == 1))
            self.assertEqual(group["ended_in_eos"], int(length < 4))
        self.assertEqual(decode.call_args.args[-1], 5)


if __name__ == "__main__":
    unittest.main()
