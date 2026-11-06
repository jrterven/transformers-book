"""Targets, loss selection, and information paths in the Chapter 7 examples."""

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from examples.ch07_pretraining.data import (
        BOS, EOS, FIRST_WORD, IGNORE, MASK, PAD, SENTINELS, VOCAB,
        bert_corrupt, causal_pairs, encode, span_corrupt, token_names,
    )
    from examples.ch07_pretraining.model import TinyTokenModel, make_model, selected_loss
    from examples.ch07_pretraining.run_examples import (
        objective_step, overfit_example, prepared_examples, visibility_probe,
    )


@unittest.skipIf(torch is None, "Chapter 7 requires PyTorch")
class PretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(71)

    def test_causal_labels_are_one_token_ahead(self):
        inputs, labels = causal_pairs(encode("the red car stopped"))
        self.assertEqual(token_names(inputs), ["the", "red", "car"])
        self.assertEqual(token_names(labels), ["red", "car", "stopped"])
        self.assertEqual(inputs.shape, labels.shape)
        with self.assertRaises(ValueError):
            causal_pairs(encode("the"))
        with self.assertRaises(TypeError):
            causal_pairs([7.0, 8.0])

    def test_mlm_labels_keep_original_word_and_ignore_other_positions(self):
        original = encode("the red car stopped")
        inputs, labels = bert_corrupt(original, [1], torch.Generator().manual_seed(7))
        self.assertEqual(token_names(inputs), ["the", "[MASK]", "car", "stopped"])
        self.assertEqual(token_names(labels), ["-", "red", "-", "-"])
        self.assertEqual(original, encode("the red car stopped"))

    def test_mlm_replacements_follow_mixture_and_never_choose_specials(self):
        # Repeated selected ordinary tokens make all three cases inspectable.
        # Random replacements may equal the original word, just as in BERT.
        original = [FIRST_WORD] * 20000
        inputs, labels = bert_corrupt(original, list(range(len(original))),
                                     torch.Generator().manual_seed(19))
        self.assertTrue(torch.equal(labels, torch.tensor(original)))
        masked_fraction = (inputs == MASK).double().mean().item()
        self.assertAlmostEqual(masked_fraction, 0.8, delta=0.015)
        self.assertTrue(((inputs == MASK) | (inputs >= FIRST_WORD)).all())
        self.assertTrue((inputs > FIRST_WORD).any())
        self.assertTrue((inputs == FIRST_WORD).any())
        again, again_labels = bert_corrupt(original, list(range(len(original))),
                                          torch.Generator().manual_seed(19))
        self.assertTrue(torch.equal(inputs, again))
        self.assertTrue(torch.equal(labels, again_labels))

    def test_mlm_rejects_missing_duplicate_and_special_prediction_positions(self):
        tokens = [BOS] + encode("the red") + [EOS, PAD]
        for selected in ([], [1, 1], [0], [3], [4], [-1], [5]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                bert_corrupt(tokens, selected)

    def test_corruption_trace_observes_without_changing_random_choices(self):
        original = encode("the red car stopped")
        plain_rng = torch.Generator().manual_seed(19)
        trace_rng = torch.Generator().manual_seed(19)
        treatments, random_same_word = set(), False
        for _ in range(128):
            plain, labels = bert_corrupt(original, [1], plain_rng)
            traced, traced_labels, trace = bert_corrupt(
                original, [1], trace_rng, return_trace=True)
            self.assertTrue(torch.equal(plain, traced))
            self.assertTrue(torch.equal(labels, traced_labels))
            self.assertTrue(torch.equal(plain_rng.get_state(), trace_rng.get_state()))
            record = trace[0]
            expected = ("mask" if record["draw"] < .8 else
                        "random" if record["draw"] < .9 else "unchanged")
            self.assertEqual(record["treatment"], expected)
            self.assertEqual(record["position"], 1)
            self.assertEqual(record["input_id"], traced[1].item())
            self.assertEqual(record["label_id"], original[1])
            treatments.add(expected)
            random_same_word |= expected == "random" and record["input_id"] == original[1]
        self.assertEqual(treatments, {"mask", "random", "unchanged"})
        self.assertTrue(random_same_word)

    def test_hidden_vectors_support_a_separate_task_head(self):
        ids = torch.tensor([encode("the red car stopped"),
                            encode("the blue car") + [PAD]])
        for causal in (True, False):
            with self.subTest(causal=causal):
                model = TinyTokenModel(causal).double().eval()
                hidden = model.encode_hidden(ids)
                self.assertEqual(hidden.shape, (2, 4, model.d_model))
                torch.testing.assert_close(model(ids), model.vocabulary_projection(hidden))
                last_valid = (ids != PAD).sum(1) - 1
                pooled = hidden[torch.arange(2), last_valid]
                head = torch.nn.Linear(model.d_model, 2).double()
                head(pooled).sum().backward()
                self.assertIsNotNone(model.embedding.weight.grad)
                self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0)

    def test_span_reconstruction_is_aligned_and_records_removed_text(self):
        original = encode("the red car stopped near the old bridge today")
        source, decoder_input, labels = span_corrupt(original, [(1, 3), (6, 8)])
        self.assertEqual(token_names(source),
                         ["the", "S0", "stopped", "near", "the", "S1", "today", "EOS"])
        self.assertEqual(token_names(labels),
                         ["S0", "red", "car", "S1", "old", "bridge", "S2", "EOS"])
        self.assertEqual(decoder_input[0].item(), BOS)
        self.assertTrue(torch.equal(decoder_input[1:], labels[:-1]))
        self.assertEqual(original, encode("the red car stopped near the old bridge today"))

    def test_span_boundaries_include_first_and_last_tokens(self):
        tokens = encode("the red car stopped")
        source, decoder, labels = span_corrupt(tokens, [(0, 1), (3, 4)])
        self.assertEqual(token_names(source), ["S0", "red", "car", "S1", "EOS"])
        self.assertEqual(token_names(labels), ["S0", "the", "S1", "stopped", "S2", "EOS"])
        self.assertEqual(decoder.shape, labels.shape)

    def test_span_builder_rejects_empty_overlapping_or_excess_spans(self):
        tokens = encode("the red car stopped")
        for spans in ([], [(1, 1)], [(-1, 1)], [(1, 5)], [(0, 2), (1, 3)],
                      [(0, 1), (1, 2)], [(2, 3), (0, 1)],
                      [(0, 1), (2, 3), (3, 4)]):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                span_corrupt(tokens, spans)

    def test_mean_loss_and_gradients_only_select_requested_label_slots(self):
        logits = torch.zeros(2, 3, len(VOCAB), dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[IGNORE, 8, IGNORE], [9, EOS, IGNORE]])
        loss = selected_loss(logits, labels)
        self.assertAlmostEqual(loss.item(), math.log(len(VOCAB)), places=12)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[labels == IGNORE],
                                    torch.zeros(3, len(VOCAB)).double()))
        # IGNORE is an index sentinel, not a vocabulary member. Ignoring a slot
        # does not remove special classes such as PAD from other softmax rows.
        self.assertGreater(logits.grad[0, 1, PAD].item(), 0.0)
        with self.assertRaises(ValueError):
            selected_loss(logits, torch.full_like(labels, IGNORE))

    def test_later_input_cannot_change_earlier_causal_predictions(self):
        original = torch.tensor([encode("the red car stopped")])
        changed = torch.tensor([encode("the red car moved")])
        causal = TinyTokenModel(True).double().eval()
        full = TinyTokenModel(False).double().eval()
        full.load_state_dict(causal.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(causal(original)[:, :3], causal(changed)[:, :3],
                                       atol=0.0, rtol=0.0)
            self.assertGreater((full(original)[:, 1] - full(changed)[:, 1]).abs().max().item(),
                               1e-6)

    def test_padding_embedding_cannot_change_real_outputs_in_either_stack(self):
        ids = torch.tensor([encode("the red car") + [PAD],
                            encode("the blue car moved")])
        for causal in (True, False):
            with self.subTest(causal=causal), torch.no_grad():
                model = TinyTokenModel(causal).double().eval()
                before = model(ids)
                model.embedding.weight[PAD].add_(1000)
                after = model(ids)
                torch.testing.assert_close(before[ids != PAD], after[ids != PAD],
                                           atol=0.0, rtol=0.0)
                # Padded query outputs may exist; ignored labels keep them out of loss.
        with self.assertRaises(ValueError):
            TinyTokenModel(True)(torch.tensor([[PAD, PAD]]))

    def test_t5_decoder_is_causal_while_encoder_source_is_available(self):
        model = make_model("t5").double().eval()
        (source, decoder), _ = prepared_examples()["t5"]
        later_decoder = decoder.clone()
        later_decoder[0, -1] = 15
        other_source = source.clone()
        other_source[0, 0] = 15
        with torch.no_grad():
            before = model(source, decoder)
            torch.testing.assert_close(before[:, :-1], model(source, later_decoder)[:, :-1],
                                       atol=0.0, rtol=0.0)
            self.assertGreater((before[:, 0] - model(other_source, decoder)[:, 0]).abs().max().item(),
                               1e-6)

    def test_all_objectives_backpropagate_finite_gradients_through_their_stacks(self):
        for family, (inputs, labels) in prepared_examples().items():
            with self.subTest(family=family):
                model = make_model(family)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                loss = objective_step(model, optimizer, inputs, labels)
                self.assertTrue(math.isfinite(loss))
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_fixed_example_training_checks_wiring_without_claiming_generalization(self):
        for family in ("gpt", "bert", "t5"):
            with self.subTest(family=family):
                result = overfit_example(family, steps=100)
                self.assertLess(result["final_nll"], 0.02 * result["initial_nll"])
                self.assertEqual(result["correct_selected"], result["selected_labels"])
                self.assertEqual(result["selected_predictions"], result["selected_targets"])
                if family == "t5":
                    self.assertTrue(result["exact_generated"])

    def test_t5_short_run_does_not_generate_padding_or_another_start(self):
        result = overfit_example("t5", steps=1)
        self.assertTrue(result["generated"])
        self.assertNotIn("PAD", result["generated"])
        self.assertNotIn("BOS", result["generated"])


if __name__ == "__main__":
    unittest.main()
