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
