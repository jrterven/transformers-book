"""Check Chapter 20's shifts, search, metrics, and optional CPU step."""

import math
import unittest

import numpy as np

from examples.ch20_generation.model import (
    EOS,
    beam_search,
    clipped_ngram_precision,
    collate_pairs,
    lcs_length,
    rouge_l,
)
from examples.ch20_generation.run_examples import (
    A,
    B,
    PAIRS,
    batch_example,
    metric_example,
    search_example,
)

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from examples.ch20_generation.training import encoder_decoder_step


class BatchTests(unittest.TestCase):
    def test_worked_batch_aligns_bos_and_eos(self):
        self.assertEqual(batch_example(), {
            "source": [[3, 4, 5], [6, 7, 0]],
            "decoder_input": [[1, 8, 9], [1, 10, 0]],
            "labels": [[8, 9, 2], [10, 2, 0]],
        })

    def test_arbitrary_unpadded_pairs_and_empty_target(self):
        pairs = (([2, 3], []), ([4], [8, 9]))
        source, decoder, labels = collate_pairs(iter(pairs), pad_id=10, bos_id=11, eos_id=12)
        np.testing.assert_array_equal(source, [[2, 3], [4, 10]])
        np.testing.assert_array_equal(decoder, [[11, 10, 10], [11, 8, 9]])
        np.testing.assert_array_equal(labels, [[12, 10, 10], [8, 9, 12]])
        self.assertEqual(labels.dtype, np.int64)
        self.assertEqual(pairs, (([2, 3], []), ([4], [8, 9])))

    def test_batch_rejects_padding_and_duplicate_boundaries(self):
        with self.assertRaisesRegex(ValueError, "unpadded"):
            collate_pairs([([4, 0], [3])])
        with self.assertRaisesRegex(ValueError, "exclude"):
            collate_pairs([([4], [3, 2])])
        with self.assertRaisesRegex(ValueError, "distinct"):
            collate_pairs(PAIRS, eos_id=0)
        with self.assertRaises(TypeError):
            collate_pairs([([3.5], [4])])


class SearchTests(unittest.TestCase):
    def test_worked_greedy_and_beam(self):
        report = search_example()
        self.assertEqual(report["greedy"]["tokens"], ["A", "EOS"])
        self.assertAlmostEqual(report["greedy"]["probability"], 0.33)
        self.assertEqual(report["beam_2"]["tokens"], ["B", "EOS"])
        self.assertAlmostEqual(report["beam_2"]["probability"], 0.38)
        self.assertTrue(report["beam_2"]["finished"])
        self.assertFalse(report["capped"]["finished"])

    def test_completed_sequence_competes_and_is_never_extended(self):
        called = []

        def next_probabilities(prefix):
            called.append(prefix)
            self.assertNotIn(EOS, prefix)
            if not prefix:
                return {A: 0.55, EOS: 0.45}
            return {A: 0.5, B: 0.5}

        tokens, logp, finished = beam_search(next_probabilities, 2, 4)
        self.assertEqual(tokens, (EOS,))
        self.assertAlmostEqual(logp, math.log(0.45))
        self.assertTrue(finished)
        self.assertGreater(len(called), 1)

    def test_ties_are_independent_of_mapping_insertion_order(self):
        for order in ({B: 0.5, A: 0.5}, {A: 0.5, B: 0.5}):
            tokens, _, finished = beam_search(lambda _: order, 1, 1)
            self.assertEqual(tokens, (A,))
            self.assertFalse(finished)

    def test_validates_callback_probabilities_and_skips_zero_mass(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            beam_search(lambda _: {A: 0.7}, 1, 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            beam_search(lambda _: {A: np.nan}, 1, 1)
        tokens, logp, finished = beam_search(lambda _: {A: 0, EOS: 1}, 2, 4)
        self.assertEqual((tokens, logp, finished), ((EOS,), 0.0, True))


class MetricTests(unittest.TestCase):
    def test_worked_clipping_and_lcs_f1(self):
        result = metric_example()
        self.assertEqual(result["clipped_unigram_precision"], 0.5)
        self.assertEqual(result["rouge_l"], {
            "lcs_length": 3, "precision": 3 / 7, "recall": 1, "f1": 0.6,
        })

    def test_clipped_repeated_ngrams_and_short_sequences(self):
        self.assertEqual(clipped_ngram_precision(["a", "b", "a", "b", "a"],
                                                 ["a", "b", "a"], 2), 0.5)
        self.assertIsNone(clipped_ngram_precision(["a"], ["a", "b"], 2))
        self.assertEqual(clipped_ngram_precision(["a"], [], 1), 0)
        with self.assertRaisesRegex(TypeError, "tokenized"):
            clipped_ngram_precision("raw string", ["a"], 1)

    def test_lcs_respects_order_and_can_skip_tokens(self):
        self.assertEqual(lcs_length(["a", "x", "b", "y", "c"], ["a", "b", "c"]), 3)
        self.assertEqual(lcs_length(["a", "b", "c"], ["c", "b", "a"]), 1)
        self.assertEqual(rouge_l([], ["a"]), {
            "lcs_length": 0, "precision": 0, "recall": 0, "f1": 0,
        })


@unittest.skipIf(torch is None, "The optional Chapter 20 update requires PyTorch")
class TrainingTests(unittest.TestCase):
    def test_encoder_decoder_step_counts_only_nonpadding_labels(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            result = encoder_decoder_step(PAIRS)
        finally:
            torch.set_num_threads(previous_threads)
        self.assertEqual(result["logit_shape"], [2, 3, 11])
        self.assertEqual(result["scored_tokens"], 5)
        self.assertTrue(math.isfinite(result["loss_before"]))
        self.assertTrue(math.isfinite(result["loss_after"]))
        self.assertLess(result["loss_after"], result["loss_before"])


if __name__ == "__main__":
    unittest.main()
