"""Compare token scores and free generation inside and outside training lengths."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import torch

from .data import VOCAB, collate, make_splits, words
from .decoding import greedy_decode
from .training import evaluate, load_checkpoint, token_loss


@torch.inference_mode()
def measure(checkpoint):
    model, metadata = load_checkpoint(checkpoint)
    settings = metadata["experiment"]
    min_length, max_length = settings["min_length"], settings["max_length"]
    if (min_length, max_length) != (3, 6):
        raise ValueError("use the recorded checkpoint trained on lengths 3 through 6")
    _, _, test = make_splits(seed=settings["data_seed"], n_train=settings["split_sizes"][0],
                             n_validation=settings["split_sizes"][1], n_test=settings["split_sizes"][2])
    _, _, longer = make_splits(seed=606, n_train=3, n_validation=3, n_test=48,
                               min_length=7, max_length=9)
    cap = 11
    groups = {}
    for length in range(3, 10):
        examples = [source for source in (test if length <= 6 else longer) if len(source) == length]
        groups[str(length)] = evaluate(model, examples, generation_max_new_tokens=cap)

    examples = [source for source in longer if len(source) == 7]
    batch = collate(examples)
    generated = greedy_decode(model, batch.source, max_new_tokens=cap, cached=False)
    # First nonmatching length-seven example in fixed data order for illustration.
    example = None
    for index, (row, target) in enumerate(zip(generated, batch.target, strict=True)):
        ids, reference = row.tolist(), target.tolist()
        if ids[:len(reference)] == reference:
            continue
        common = next(k for k, answer in enumerate(reference) if ids[k] != answer)
        one = collate([examples[index]])
        logits = model(one.source, one.decoder_input)
        p = logits.double().softmax(-1).gather(-1, one.target[..., None])[0, :, 0]
        total, count = token_loss(logits, one.target)
        example = {"index_in_group": index + 1, "source": words(one.source[0]),
                   "reference": words(reference, include_special=True),
                   "generated": words(ids, include_special=True),
                   "first_difference": common + 1, "expected": VOCAB[reference[common]],
                   "chosen": VOCAB[ids[common]], "reference_token_probabilities": p.tolist(),
                   "mean_nll": float(total / count)}
        break
    with checkpoint.open("rb") as f:
        digest = file_digest(f, "sha256").hexdigest()
    return {"selected_update": metadata["step"], "checkpoint_sha256": digest,
            "torch": torch.__version__, "seed_for_longer_sequences": 606,
            "generation_cap": cap, "groups": groups, "example": example}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = measure(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
