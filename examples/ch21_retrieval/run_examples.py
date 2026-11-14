"""Run the Chapter 21 retrieval and evidence checks locally."""

import argparse
import json
from pathlib import Path

from .model import (
    bm25_scores,
    dense_scores,
    evidence_recall,
    literal_evidence_hits,
    pack_context,
    rank_passages,
    rerank_candidates,
    tokenize,
)


CORPUS = {
    "P1": "Renew a parking permit at the mobility desk.",
    "P2": "A parking permit expires on June 30.",
    "P3": "The recycling office opens on Tuesday.",
}
QUERY = "Where can I extend my parking authorization?"


def retrieval_example():
    ids = tuple(CORPUS)
    passages = tuple(CORPUS.values())
    lexical = bm25_scores("parking permit", passages)
    paraphrase = bm25_scores(QUERY, passages)
    dense = dense_scores([1, 0], [[0.8, 0.6], [0.6, 0.8], [0, 1]])
    lexical_rank = rank_passages(ids, lexical)
    dense_rank = rank_passages(ids, dense)
    candidate_ids = lexical_rank[:2]
    pair_scores = {"P1": 0.9, "P2": 0.2}
    reranked = rerank_candidates(candidate_ids, pair_scores)
    budget = 10
    packed = pack_context(reranked, CORPUS, budget)
    relevant = ("P1",)
    return {
        "passage_ids": ids,
        "passage_word_counts": [len(tokenize(text)) for text in passages],
        "lexical_scores": lexical.tolist(),
        "lexical_rank": lexical_rank,
        "paraphrase_query": QUERY,
        "paraphrase_lexical_scores": paraphrase.tolist(),
        "dense_scores": dense.tolist(),
        "dense_rank": dense_rank,
        "candidate_ids": candidate_ids,
        "provided_paired_scores": pair_scores,
        "reranked": reranked,
        "context_budget": budget,
        "packed_passages": packed,
        "packed_word_count": sum(len(tokenize(text)) for _, text in packed),
        "literal_mobility_desk_hits": literal_evidence_hits(packed, "mobility desk"),
        "answer_fixture": {"text": "At the mobility desk.", "evidence_ids": relevant},
        "lexical_recall_at_1": evidence_recall(lexical_rank, relevant, 1),
        "lexical_recall_at_2": evidence_recall(lexical_rank, relevant, 2),
        "reranked_recall_at_1": evidence_recall(reranked, relevant, 1),
    }


def no_price_example():
    query = "How much does a parking permit cost?"
    rank = rank_passages(CORPUS, bm25_scores(query, CORPUS.values()), limit=2)
    packed = pack_context(rank, CORPUS, token_budget=10)
    return {
        "query": query,
        "retrieved_ids": rank,
        "packed_ids": tuple(pid for pid, _ in packed),
        "literal_permit_price_hits": literal_evidence_hits(packed, "permit price"),
        "fixture_gold_evidence_ids": (),
        "answer_fixture": {"text": None, "evidence_ids": ()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {"retrieval": retrieval_example(), "permit_price": no_price_example()}
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
