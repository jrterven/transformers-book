"""Small NumPy reference functions for ranking and packing passages."""

from collections import Counter
from collections.abc import Mapping
import math
from operator import index
import re

import numpy as np


def tokenize(text):
    """Split text into lowercase word tokens, including numbers."""
    if not isinstance(text, str):
        raise TypeError("tokenization needs text")
    return tuple(re.findall(r"[^\W_]+", text.casefold()))


# snippet:bm25_scores
def bm25_scores(query, passage_texts, *, k1=1.2, b=0.75):
    """Return BM25 scores in passage order using positive IDF."""
    if not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("k1 must be positive and finite, and b between zero and one")
    if isinstance(passage_texts, (str, bytes)):
        raise TypeError("pass a sequence of passage strings")
    documents = [Counter(tokenize(text)) for text in passage_texts]
    lengths = np.array([sum(doc.values()) for doc in documents], dtype=np.float64)
    if not lengths.size or not lengths.any():
        raise ValueError("BM25 needs a corpus with at least one word")

    length_term = k1 * (1 - b + b * lengths / lengths.mean())
    scores = np.zeros(len(documents))
    for term in dict.fromkeys(tokenize(query)):
        frequency = np.array([doc[term] for doc in documents], dtype=np.float64)
        present = frequency > 0
        df = int(present.sum())
        if df == 0:
            continue
        idf = math.log1p((len(documents) - df + 0.5) / (df + 0.5))
        tf = frequency[present]
        scores[present] += idf * (k1 + 1) * tf / (tf + length_term[present])
    return scores
# end-snippet:bm25_scores


def _passage_ids(passage_ids):
    if isinstance(passage_ids, (str, bytes)):
        raise TypeError("pass a sequence of passage IDs")
    ids = tuple(passage_ids)
    if any(not isinstance(pid, str) or not pid for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("passage IDs must be unique nonempty strings")
    return ids


def _real_array(array, name):
    array = np.asarray(array)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    return array.astype(np.float64, copy=False)


# snippet:dense_scores
def dense_scores(query_vector, passage_vectors):
    """Compute raw dot products in the supplied passage order."""
    query = _real_array(query_vector, "query vector")
    passages = _real_array(passage_vectors, "passage vectors")
    if (query.ndim != 1 or query.size == 0 or passages.ndim != 2
            or passages.shape[1] != query.size):
        raise ValueError("use a query vector and a passage matrix of the same width")
    if not (np.isfinite(query).all() and np.isfinite(passages).all()):
        raise ValueError("query and passage vectors must be finite")
    with np.errstate(over="raise", invalid="raise"):
        return passages @ query
# end-snippet:dense_scores


# snippet:rank_passages
def rank_passages(passage_ids, scores, limit=None):
    """Rank higher scores first and break ties by ascending passage ID."""
    ids = _passage_ids(passage_ids)
    scores = np.asarray(scores)
    if (scores.shape != (len(ids),) or scores.dtype.kind not in "iuf"
            or not np.isfinite(scores).all()):
        raise ValueError("supply one finite real score per passage ID")
    if limit is not None:
        limit = index(limit)
        if limit < 0:
            raise ValueError("the rank limit must be nonnegative")
    by_id = sorted(zip(ids, scores))
    ranked = sorted(by_id, key=lambda pair: pair[1], reverse=True)
    return tuple(pid for pid, _ in ranked[:limit])


def rerank_candidates(candidate_ids, paired_scores):
    """Order only the candidates using scores keyed by their IDs."""
    ids = _passage_ids(candidate_ids)
    if not isinstance(paired_scores, Mapping) or set(paired_scores) != set(ids):
        raise ValueError("supply exactly one paired score per candidate ID")
    return rank_passages(ids, [paired_scores[pid] for pid in ids])
# end-snippet:rank_passages


# snippet:pack_context
def pack_context(ranked_ids, corpus, token_budget):
    """Pack whole passages, preserving IDs and the rank order of kept items."""
    ids = _passage_ids(ranked_ids)
    token_budget = index(token_budget)
    if token_budget < 0:
        raise ValueError("token budget must be nonnegative")
    packed = []
    for pid in ids:
        text = corpus[pid]
        cost = len(tokenize(text))
        if cost <= token_budget:
            packed.append((pid, text))
            token_budget -= cost
    return tuple(packed)
# end-snippet:pack_context


# snippet:evidence_occurrence
def literal_evidence_hits(context, phrase):
    """Return IDs that contain the consecutive word tokens of the phrase."""
    wanted = tokenize(phrase)
    if not wanted:
        raise ValueError("the evidence phrase needs at least one word")
    hits = []
    for pid, text in context:
        words = tokenize(text)
        if any(words[i:i + len(wanted)] == wanted
               for i in range(len(words) - len(wanted) + 1)):
            hits.append(pid)
    return tuple(hits)


def evidence_recall(ranked_ids, relevant_ids, k):
    """Return the fraction of supplied gold evidence IDs retrieved by rank k."""
    retrieved = _passage_ids(ranked_ids)
    relevant = set(_passage_ids(relevant_ids))
    k = index(k)
    if k < 0:
        raise ValueError("k must be nonnegative")
    return len(relevant.intersection(retrieved[:k])) / len(relevant) if relevant else None
# end-snippet:evidence_occurrence
