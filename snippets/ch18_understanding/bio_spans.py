def bio_spans(tags, *, repair=False):
    """Read BIO spans as (start, exclusive end, entity type) triples."""
    spans = set()
    start, entity = 0, None
    # A final O closes a span that reaches the end of the sentence.
    for position, tag in enumerate((*tags, "O")):
        if not isinstance(tag, str):
            raise ValueError("BIO tags must be strings")
        prefix, _, label = tag.partition("-")
        if tag != "O":
            if prefix not in ("B", "I") or not label:
                raise ValueError(f"invalid BIO tag at word {position}: {tag}")
            if prefix == "I":
                if label == entity:
                    continue
                if not repair:
                    raise ValueError(f"unexpected inside tag at word {position}: {tag}")

        if entity is not None:
            spans.add((start, position, entity))
        start = position
        entity = None if tag == "O" else label
    return spans
