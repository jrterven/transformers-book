"""Probability accounting checks; no model download or optional dependencies."""

import unittest
import numpy as np

from examples.ch01_language.pretrained_figures import candidate_distribution


class CandidateDistributionTests(unittest.TestCase):
    def test_selected_tokens_keep_full_vocabulary_denominator(self):
        selected, remaining = candidate_distribution(np.log([2., 1., 3.]), [0, 2])
        np.testing.assert_allclose(selected, [1 / 3, 1 / 2])
        self.assertAlmostEqual(remaining, 1 / 6)
        self.assertAlmostEqual(selected.sum() + remaining, 1)

    def test_common_logit_offset_does_not_change_mass(self):
        expected = candidate_distribution([2., 1., 0., -1.], [0, 2])
        actual = candidate_distribution([1002., 1001., 1000., 999.], [0, 2])
        np.testing.assert_allclose(actual[0], expected[0])
        self.assertAlmostEqual(actual[1], expected[1])

    def test_all_entries_leave_no_residual(self):
        selected, remaining = candidate_distribution([0., 0.], [1, 0])
        np.testing.assert_allclose(selected, [.5, .5])
        self.assertEqual(remaining, 0)

    def test_invalid_candidates_and_logits(self):
        for ids in ([0, 0], [-1], [2], [0.5], [], [[0]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                candidate_distribution([0., 1.], ids)
        for z in ([], [[0., 1.]], [0., np.nan], [np.inf]):
            with self.subTest(logits=z), self.assertRaises(ValueError):
                candidate_distribution(z, [0])


if __name__ == "__main__":
    unittest.main()
