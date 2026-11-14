"""Run Chapter 20's batch, search, and reference-metric examples."""

import argparse
import json
import math
from pathlib import Path

from .model import EOS, beam_search, clipped_ngram_precision, collate_pairs, rouge_l


PAIRS = (((3, 4, 5), (8, 9)), ((6, 7), (10,)))
A, B, C = 3, 4, 5


def toy_probabilities(prefix):
    distributions = {
        (): {A: 0.6, B: 0.4},
        (A,): {EOS: 0.55, C: 0.45},
        (B,): {EOS: 0.95, C: 0.05},
        (A, C): {EOS: 1.0},
        (B, C): {EOS: 1.0},
    }
    return distributions[prefix]


def batch_example():
    source, decoder_input, labels = collate_pairs(PAIRS)
    return {
        "source": source.tolist(),
        "decoder_input": decoder_input.tolist(),
        "labels": labels.tolist(),
    }


def search_example():
    names = {A: "A", B: "B", C: "C", EOS: "EOS"}
    result = {}
    for label, width, cap in (("greedy", 1, 3), ("beam_2", 2, 3), ("capped", 2, 1)):
        tokens, log_probability, finished = beam_search(toy_probabilities, width, cap)
        result[label] = {
            "tokens": [names[token] for token in tokens],
            "log_probability": log_probability,
            "probability": math.exp(log_probability),
            "finished": finished,
        }
    return result


def metric_example():
    candidate = ("bus", "bus", "bus", "late")
    reference = ("the", "blue", "bus", "arrived", "late")
    summary = ("the", "bus", "arrived", "late", "due", "to", "snow")
    gold = ("bus", "arrived", "late")
    return {
        "clipped_unigram_precision": clipped_ngram_precision(candidate, reference, 1),
        "rouge_l": rouge_l(summary, gold),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    parser.add_argument("--train-step", action="store_true", help="run the optional PyTorch step")
    args = parser.parse_args()
    result = {"batch": batch_example(), "search": search_example(), "metrics": metric_example()}
    if args.train_step:
        import torch

        from .training import encoder_decoder_step

        torch.set_num_threads(1)
        result["encoder_decoder_step"] = encoder_decoder_step(PAIRS)
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
