"""Run Chapter 23's image-text scores, retrieval, and bridge locally."""

import argparse
import json
import math
from pathlib import Path

from examples.ch04_attention.attention import stable_softmax

from .model import (
    bidirectional_recall_at_k,
    contrastive_logits,
    image_feature_bridge,
    symmetric_contrastive_loss,
)


def contrastive_example():
    logits = contrastive_logits([[1, 0], [0, 1]], [[1, 0], [0.6, 0.8]], scale=2)
    image_loss, text_loss, mean = symmetric_contrastive_loss(logits)
    return {
        "logits": logits.tolist(),
        "text_candidate_weights_by_image": stable_softmax(logits).tolist(),
        "image_to_text_loss": image_loss,
        "text_to_image_loss": text_loss,
        "mean_loss": mean,
    }


def retrieval_example():
    scores = [[2, 1.2, -1], [0, 1.6, -1], [-1, -1, -1]]
    relevance = [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
    return {
        "scores": scores,
        "relevance": relevance,
        "recall_at_1": bidirectional_recall_at_k(scores, relevance, 1),
        "recall_at_2": bidirectional_recall_at_k(scores, relevance, 2),
    }


def bridge_example():
    queries = [[math.sqrt(2), 0]]
    keys = [[0, 0], [math.log(3), 0]]
    values = [[2, 0], [0, 4]]
    projection = [[1, 0, 1], [0, 1, -1]]
    context, weights, language_tokens = image_feature_bridge(
        queries, keys, values, projection
    )
    return {
        "context": context.tolist(),
        "attention_weights": weights.tolist(),
        "language_width_tokens": language_tokens.tolist(),
        "token_shape": list(language_tokens.shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "contrastive": contrastive_example(),
        "retrieval": retrieval_example(),
        "visual_bridge": bridge_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
