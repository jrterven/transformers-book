"""Checks of learning, evaluation, and incremental execution in Chapter 6."""

import math
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from examples.ch05_transformer.model import Transformer
    from examples.ch06_training.data import BOS, EOS, PAD, VOCAB, collate, make_splits
    from examples.ch06_training.decoding import (
        CachedDecoder, benchmark, decode_prefix, encode_source, greedy_decode,
    )
    from examples.ch06_training.training import (
        evaluate, exact_matches, load_checkpoint, save_checkpoint, token_loss, train_step,
    )
    from examples.ch06_training.run_experiment import loss_sanity_example, run


@unittest.skipIf(torch is None, "Chapter 6 requires PyTorch")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(23)
        self.config = dict(source_vocab_size=9, target_vocab_size=9, d_model=8,
                           n_heads=2, d_ff=16, n_encoder_layers=2,
                           n_decoder_layers=2, dropout=0.0, max_length=32)
        self.model = Transformer(**self.config).double().eval()
        self.batch = collate([(3, 4, 5, 6), (6, 4, 3)])

    def test_splits_are_disjoint_unique_balanced_and_reproducible(self):
        splits = make_splits()
        self.assertEqual(splits, make_splits())
        self.assertNotEqual(splits, make_splits(seed=4))
        sets = list(map(set, splits))
        for values, unique, size in zip(splits, sets, (512, 64, 64)):
            self.assertEqual(len(unique), size)
            self.assertEqual({n: sum(len(s) == n for s in values) for n in range(3, 7)},
                             {n: size // 4 for n in range(3, 7)})
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_shift_has_bos_inputs_eos_targets_and_right_padding(self):
        self.assertEqual(self.batch.source.tolist(), [[3, 4, 5, 6], [6, 4, 3, 0]])
        self.assertEqual(self.batch.decoder_input.tolist(),
                         [[BOS, 6, 5, 4, 3], [BOS, 3, 4, 6, PAD]])
        self.assertEqual(self.batch.target.tolist(),
                         [[6, 5, 4, 3, EOS], [3, 4, 6, EOS, PAD]])
        self.assertEqual(int((self.batch.target != PAD).sum()), 9)

    def test_longer_sequences_can_be_sampled_without_enumerating_every_combination(self):
        # There are over two billion possible length-12 color sequences.
        # A small teaching dataset should only need memory for its samples.
        splits = make_splits(n_train=4, n_validation=2, n_test=2,
                             min_length=12, max_length=12)
        values = [sequence for split in splits for sequence in split]
        self.assertEqual(len(values), 8)
        self.assertEqual(len(set(values)), 8)
        self.assertTrue(all(len(sequence) == 12 for sequence in values))
        self.assertTrue(all(3 <= token < len(VOCAB)
                            for sequence in values for token in sequence))

    def test_loss_uses_raw_logits_and_ignores_only_padding_positions(self):
        logits = torch.zeros(2, 3, 9, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[3, EOS, PAD], [4, 5, EOS]])
        total, count = token_loss(logits, targets)
        self.assertEqual(count.item(), 5)
        self.assertAlmostEqual(total.item(), 5 * math.log(9), places=12)
        (total / count).backward()
        self.assertTrue(torch.equal(logits.grad[0, 2], torch.zeros(9).double()))
        # PAD's class still participates in each valid softmax denominator.
        self.assertGreater(logits.grad[0, 0, PAD].item(), 0)
        with self.assertRaises(ValueError):
            token_loss(logits, torch.zeros_like(targets))

    def test_worked_loss_sanity_counts_eos_and_excludes_one_padded_label(self):
        result = loss_sanity_example()
        # Three colors plus EOS, then two colors plus EOS: seven labels.
        # Every reference receives probability 1/9 in the uniform vocabulary.
        self.assertEqual(result["tokens"], 7)
        self.assertAlmostEqual(result["loss_sum"], 7 * math.log(9), places=12)
        self.assertAlmostEqual(result["nll"], math.log(9), places=12)
        self.assertAlmostEqual(result["perplexity"], 9.0, places=12)

    def test_evaluation_is_token_weighted_and_batch_partition_invariant(self):
        data = [(3, 4, 5), (4, 4, 6, 6, 5, 3), (5, 3, 6, 4)]
        whole = evaluate(self.model, data, batch_size=3)
        divided = evaluate(self.model, data, batch_size=1)
        self.assertAlmostEqual(whole["nll"], divided["nll"], places=12)
        self.assertAlmostEqual(whole["perplexity"], math.exp(whole["nll"]), places=12)
        self.assertEqual(whole["tokens"], 16)
        self.assertEqual(whole["exact_match"], divided["exact_match"])

    def test_termination_counts_eos_but_not_padding_or_length_cap(self):
        data = [(3, 4, 5), (5, 4, 3)]
        with patch("examples.ch06_training.training.greedy_decode",
                   return_value=torch.tensor([[EOS, PAD, PAD], [4, 4, 4]])):
            result = evaluate(self.model, data)
        self.assertEqual(result["termination_rate"], 0.5)
        self.assertEqual(result["exact_match"], 0.0)

    def test_termination_uses_the_same_generation_cap_across_batch_partitions(self):
        data = [(3, 4, 5), (5, 4, 3, 6, 7, 8)]
        def delayed_eos(model, source, max_new_tokens):
            result = source.new_full((source.shape[0], max_new_tokens), 4)
            if max_new_tokens >= 7:
                result[:, 6] = EOS
                result[:, 7:] = PAD
            return result
        with patch("examples.ch06_training.training.greedy_decode",
                   side_effect=delayed_eos):
            together = evaluate(self.model, data, batch_size=2)
            separate = evaluate(self.model, data, batch_size=1)
            shorter_cap = evaluate(self.model, data, batch_size=1,
                                   generation_max_new_tokens=6)
        self.assertEqual(together["termination_rate"], 1.0)
        self.assertEqual(separate["termination_rate"], together["termination_rate"])
        self.assertEqual(shorter_cap["termination_rate"], 0.0)

    def test_default_generation_cap_respects_the_model_position_limit(self):
        # A source of length L-1 leaves exactly L decoder positions for EOS.
        # An optional extra generation step must not make this valid case fail.
        source_length = self.config["max_length"] - 1
        result = evaluate(self.model, [(3,) * source_length])
        self.assertEqual(result["tokens"], source_length + 1)
        self.assertEqual(result["sequences"], 1)
        self.assertTrue(math.isfinite(result["nll"]))

    def test_cached_logits_match_each_full_prefix_at_nonzero_positions(self):
        with torch.no_grad():
            memory, allowed = encode_source(self.model, self.batch.source)
            cache = CachedDecoder(self.model, memory, allowed)
            prefix = self.batch.decoder_input.clone()
            prefix[prefix == PAD] = EOS
            for t in range(prefix.shape[1]):
                cached = cache.step(prefix[:, t:t + 1])
                full = decode_prefix(self.model, prefix[:, :t + 1], memory, allowed)[:, -1]
                torch.testing.assert_close(cached, full, atol=1e-12, rtol=1e-12)
                self.assertEqual(cache.position, t + 1)
                for k, v in cache.self_kv:
                    self.assertEqual(k.shape, (2, 2, t + 1, 4))
                    self.assertEqual(v.shape, k.shape)
            self.assertNotEqual(cache.self_kv[0][0].data_ptr(),
                                cache.self_kv[1][0].data_ptr())

    def test_source_encoding_and_full_prefix_match_original_forward(self):
        with torch.no_grad():
            memory, allowed = encode_source(self.model, self.batch.source)
            prefix = self.batch.decoder_input.clone()
            prefix[prefix == PAD] = EOS
            full = decode_prefix(self.model, prefix, memory, allowed)
            original = self.model(self.batch.source, prefix)
        torch.testing.assert_close(full, original, atol=1e-12, rtol=1e-12)

    def test_static_cross_cache_excludes_padding(self):
        with torch.no_grad():
            memory, allowed = encode_source(self.model, self.batch.source)
            perturbed = memory.clone()
            perturbed[1, -1] = 1000 * torch.randn_like(perturbed[1, -1])
            a = CachedDecoder(self.model, memory, allowed)
            b = CachedDecoder(self.model, perturbed, allowed)
            cross_storage = [kv[0].data_ptr() for kv in a.cross_kv]
            for token in (BOS, 3, 4):
                inputs = torch.full((2, 1), token)
                torch.testing.assert_close(a.step(inputs), b.step(inputs), atol=1e-12, rtol=1e-12)
            self.assertEqual(cross_storage, [kv[0].data_ptr() for kv in a.cross_kv])

    def test_greedy_cache_and_full_prefix_generate_the_same_ids(self):
        cached = greedy_decode(self.model, self.batch.source, cached=True)
        full = greedy_decode(self.model, self.batch.source, cached=False)
        self.assertTrue(torch.equal(cached, full))
        self.assertFalse((cached == BOS).any())

    def test_eos_stops_each_row_and_length_cap_is_not_success(self):
        # Different rows finish at different times; PAD cannot be emitted early.
        calls = []
        def fake_prefix(model, prefix, memory, allowed):
            step = prefix.shape[1] - 1
            calls.append(step)
            logits = torch.full((2, prefix.shape[1], 9), -100.0)
            logits[0, -1, EOS] = 10
            logits[1, -1, EOS if step == 2 else 4] = 10
            return logits
        with patch("examples.ch06_training.decoding.decode_prefix", side_effect=fake_prefix):
            generated = greedy_decode(self.model, self.batch.source, cached=False)
        self.assertEqual(generated.tolist(), [[EOS, PAD, PAD], [4, 4, EOS]])
        self.assertEqual(calls, [0, 1, 2])
        targets = torch.tensor([[EOS, PAD, PAD], [4, 4, EOS]])
        self.assertEqual(exact_matches(generated, targets), 2)
        self.assertEqual(exact_matches(torch.tensor([[4, 4]]), targets[1:]), 0)
        self.assertEqual(exact_matches(torch.tensor([[4, PAD, 4, EOS]]), targets[1:]), 0)
        self.assertEqual(exact_matches(torch.tensor([[4, 4]]), torch.tensor([[4, 4]])), 0)

    def test_cached_generation_immediate_eos_and_no_eos_cap(self):
        with torch.no_grad():
            self.model.vocabulary_projection.weight.zero_()
            self.model.vocabulary_projection.bias.fill_(-100)
            self.model.vocabulary_projection.bias[EOS] = 100
        for cached in (False, True):
            got = greedy_decode(self.model, self.batch.source, cached=cached)
            self.assertEqual(got.tolist(), [[EOS], [EOS]])
        with torch.no_grad():
            self.model.vocabulary_projection.bias[EOS] = -100
            self.model.vocabulary_projection.bias[4] = 100
        for cached in (False, True):
            got = greedy_decode(self.model, self.batch.source, max_new_tokens=3,
                                cached=cached)
            self.assertEqual(got.tolist(), [[4, 4, 4], [4, 4, 4]])
            self.assertEqual(exact_matches(got, self.batch.target), 0)

    def test_generation_requires_eval_and_cache_rejects_pad_and_overflow(self):
        self.model.train()
        with self.assertRaises(ValueError):
            greedy_decode(self.model, self.batch.source)
        self.model.eval()
        with torch.no_grad():
            memory, allowed = encode_source(self.model, self.batch.source)
            cache = CachedDecoder(self.model, memory, allowed)
            with self.assertRaises(ValueError):
                cache.step(torch.zeros(2, 1, dtype=torch.long))
            cache.position = self.config["max_length"]
            with self.assertRaises(ValueError):
                cache.step(torch.ones(2, 1, dtype=torch.long))

    def test_checkpoint_preserves_a_snapshot_and_metadata(self):
        batch = self.batch
        before = self.model(batch.source, batch.decoder_input).detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, self.model, self.config, {"seed": 23}, 7, {"nll": 0.25})
            with torch.no_grad():
                next(self.model.parameters()).add_(1)
            restored, metadata = load_checkpoint(path)
            got = restored(batch.source, batch.decoder_input)
            self.assertEqual(next(restored.parameters()).dtype, torch.float64)
            torch.testing.assert_close(got, before, atol=1e-12, rtol=1e-12)
            self.assertEqual(metadata["step"], 7)
            self.assertEqual(metadata["experiment"]["seed"], 23)
            self.assertEqual(metadata["vocabulary"], list(VOCAB))
            self.assertFalse(restored.training)

    def test_a_real_training_step_reduces_loss_on_a_tiny_fixed_task(self):
        model = Transformer(9, 9, d_model=16, n_heads=2, d_ff=32,
                            dropout=0.0, max_length=16)
        batch = collate([(3, 4, 5), (5, 4, 3), (6, 3, 4), (4, 6, 5)])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        losses = [train_step(model, optimizer, batch) for _ in range(80)]
        self.assertLess(losses[-1], 0.1 * losses[0])
        model.eval()
        result = greedy_decode(model, batch.source)
        self.assertEqual(exact_matches(result, batch.target), 4)

    def test_cpu_benchmark_reports_fixed_work_and_finite_timings(self):
        result = benchmark(self.model, self.batch.source[:1],
                           self.batch.decoder_input[:1, :3], repeats=2)
        self.assertEqual(result["prefix_tokens"], 3)
        self.assertEqual(result["batch_size"], 1)
        self.assertEqual(result["dtype"], "torch.float64")
        self.assertEqual(result["device"], "cpu")
        for key, value in result.items():
            if key.endswith("_ms"):
                self.assertTrue(math.isfinite(value) and value > 0)

    def test_runner_keeps_one_generation_cap_for_whole_and_per_length_scores(self):
        # Shorter groups must not receive a different stopping allowance.
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            with patch("examples.ch06_training.run_experiment.evaluate",
                       wraps=evaluate) as scoring:
                result = run(directory, steps=1, validation_every=1)
        self.assertEqual(result["experiment"]["generation_max_new_tokens"], 8)
        self.assertEqual(len(result["test_by_length"]), 4)
        self.assertGreaterEqual(scoring.call_count, 7)
        self.assertTrue(all(call.kwargs["generation_max_new_tokens"] == 8
                            for call in scoring.call_args_list))


if __name__ == "__main__":
    unittest.main()
