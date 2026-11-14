"""Run Chapter 19's offline sentence and entity examples."""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import (
    bio_spans,
    first_piece_labels,
    first_piece_predictions,
    sentence_logits,
    span_metrics,
)


def pooling_example():
    features = [[1, 0], [0, 2], [2, 1], [9, 9]]
    valid = [True, True, True, False]
    weight = [[1, -1], [0.5, 1]]
    bias = [0, -0.25]
    pooled, logits = sentence_logits(features, valid, weight, bias)
    return {"pooled": pooled.tolist(), "logits": logits.tolist()}


def entity_example():
    words = ("Ari", "moved", "to", "New", "Harbor", ".")
    tokens = ("[CLS]", "Ari", "moved", "to", "New", "Har", "##bor", ".", "[SEP]", "[PAD]")
    word_ids = [-1, 0, 1, 2, 3, 4, 4, 5, -1, -1]
    tags = ("O", "B-PER", "B-LOC", "I-LOC")
    word_labels = [1, 0, 0, 2, 3, 0]
    token_predictions = [0, 1, 0, 0, 2, 0, 3, 0, 0, 0]
    labels = first_piece_labels(word_ids, word_labels)
    predicted_ids = first_piece_predictions(word_ids, token_predictions)
    gold = [tags[i] for i in word_labels]
    predicted = [tags[i] for i in predicted_ids]
    return {
        "words": words,
        "tokens": tokens,
        "word_ids": word_ids,
        "token_labels": labels.tolist(),
        "token_predictions": token_predictions,
        "gold_tags": gold,
        "predicted_tags": predicted,
        "gold_spans": sorted(bio_spans(gold)),
        "predicted_spans": sorted(bio_spans(predicted)),
        "metrics": span_metrics(gold, predicted),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {"sentence": pooling_example(), "entities": entity_example()}
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
