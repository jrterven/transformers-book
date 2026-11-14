"""Check Chapter 21's scores, stable ranks, packing, and evidence IDs."""

import math
import unittest

import numpy as np

from examples.ch21_retrieval.model import (
    bm25_scores,
    dense_scores,
    evidence_recall,
    literal_evidence_hits,
    pack_context,
    rank_passages,
    rerank_candidates,
    tokenize,
)
from examples.ch21_retrieval.run_examples import CORPUS, no_price_example, retrieval_example


class RankingTests(unittest.TestCase):
    def test_worked_bm25_scores_and_length_normalization(self):
        scores = bm25_scores("parking permit", CORPUS.values())
        idf = math.log(1.6)
        expected = [2 * idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 8 / 7)), 2 * idf, 0]
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-15)
        self.assertEqual(rank_passages(CORPUS, scores), ("P2", "P1", "P3"))
        np.testing.assert_array_equal(bm25_scores("PARKING permit permit", CORPUS.values()),
                                      scores)

    def test_bm25_term_frequency_has_diminishing_gain(self):
        texts = ("bus", "bus bus", "bus bus bus")
        scores = bm25_scores("bus", texts, b=0)
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[2], scores[1])
        self.assertLess(scores[2] - scores[1], scores[1] - scores[0])
        self.assertEqual(tokenize("PERMIT—30!"), ("permit", "30"))
        np.testing.assert_array_equal(bm25_scores("missing", texts), 0)
        with self.assertRaisesRegex(ValueError, "at least one word"):
            bm25_scores("bus", ["", "!!"])

    def test_dense_dot_products_preserve_scale_and_id_order(self):
        scores = dense_scores([2, 0], [[3, 4], [-1, 0], [0, 1]])
        np.testing.assert_array_equal(scores, [6, -2, 0])
        self.assertEqual(rank_passages(["P2", "P1", "P3"], scores), ("P2", "P3", "P1"))
        with self.assertRaisesRegex(ValueError, "same width"):
            dense_scores([1, 0], [[1, 0, 0]])
        with self.assertRaisesRegex(ValueError, "real numbers"):
            dense_scores([1j, 0], [[1, 0]])

    def test_rank_ties_and_large_integer_order(self):
        self.assertEqual(rank_passages(["P2", "P1", "P3"], [1, 1, 0]), ("P1", "P2", "P3"))
        large = np.array([2**53, 2**53 + 1], dtype=np.int64)
        self.assertEqual(rank_passages(["P1", "P2"], large), ("P2", "P1"))
        self.assertEqual(rank_passages(["P1"], [0], limit=0), ())
        with self.assertRaisesRegex(ValueError, "one finite real score"):
            rank_passages(["P1", "P2"], [1, np.nan])
        with self.assertRaisesRegex(ValueError, "unique"):
            rank_passages(["P1", "P1"], [1, 2])

    def test_reranking_aligns_scores_by_id_and_rejects_extra_candidates(self):
        paired = {"P1": 0.9, "P2": 0.2}
        self.assertEqual(rerank_candidates(["P2", "P1"], paired), ("P1", "P2"))
        with self.assertRaisesRegex(ValueError, "per candidate"):
            rerank_candidates(["P2"], paired)
        with self.assertRaisesRegex(ValueError, "per candidate"):
            rerank_candidates(["P2", "P3"], paired)


class EvidenceTests(unittest.TestCase):
    def test_packing_keeps_whole_passages_under_a_word_budget(self):
        corpus = {"a": "four words in total", "b": "two words", "c": "one"}
        packed = pack_context(["a", "b", "c"], corpus, 3)
        self.assertEqual(packed, (("b", "two words"), ("c", "one")))
        self.assertEqual(sum(len(tokenize(text)) for _, text in packed), 3)
        self.assertEqual(pack_context(["a", "b"], corpus, 4), (("a", corpus["a"]),))
        with self.assertRaises(KeyError):
            pack_context(["missing"], corpus, 0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            pack_context(["a"], corpus, -1)

    def test_literal_matches_stay_within_passage_and_token_boundaries(self):
        context = (("a", "mobility"), ("b", "desk"), ("c", "MOBILITY, Desk."))
        self.assertEqual(literal_evidence_hits(context, "mobility desk"), ("c",))
        self.assertEqual(literal_evidence_hits(context, "des"), ())
        self.assertEqual(literal_evidence_hits(context, "desk mobility"), ())
        with self.assertRaisesRegex(ValueError, "at least one word"):
            literal_evidence_hits(context, "?!")

    def test_recall_counts_gold_ids_once_and_handles_empty_gold(self):
        self.assertEqual(evidence_recall(["P2", "P1", "P3"], ["P1", "P3"], 2), 0.5)
        self.assertEqual(evidence_recall(["P1"], ["P1"], 0), 0)
        self.assertIsNone(evidence_recall(["P1"], [], 1))
        with self.assertRaisesRegex(ValueError, "unique"):
            evidence_recall(["P1", "P1"], ["P1"], 2)

    def test_retrieval_pipeline_preserves_evidence_and_missing_price_fixture(self):
        report = retrieval_example()
        self.assertEqual(report["candidate_ids"], ("P2", "P1"))
        self.assertEqual(report["reranked"], ("P1", "P2"))
        self.assertEqual(report["dense_rank"], ("P1", "P2", "P3"))
        self.assertEqual(report["packed_word_count"], 8)
        self.assertEqual(report["literal_mobility_desk_hits"], ("P1",))
        self.assertEqual(report["lexical_recall_at_1"], 0)
        self.assertEqual(report["lexical_recall_at_2"], 1)
        self.assertEqual(report["reranked_recall_at_1"], 1)
        missing = no_price_example()
        self.assertEqual(missing["literal_permit_price_hits"], ())
        self.assertIsNone(missing["answer_fixture"]["text"])


if __name__ == "__main__":
    unittest.main()
