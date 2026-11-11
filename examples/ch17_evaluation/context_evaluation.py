"""Measure evidence placement with a fixed SmolLM2 delivery-register probe."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np


SEED = 17
REPORTS = 8
RECORD_COUNTS = (16, 64, 256)
POSITIONS = ("early", "middle", "late")
COLORS = ("red", "blue")
SUFFIX_RECORDS = 8
HEADER = (
    "Read the delivery register. Each account has one code.\n\n"
    "Example:\nAccount 517: red\nAccount 631: blue\nCode for account 631: blue"
    "\n\nRegister"
)


def register_cases():
    """Yield matched placements, code flips, fixed suffixes, and a missing row."""
    rng = np.random.default_rng(SEED)
    for report in range(REPORTS):
        key = 9000 + report
        ids = rng.choice(np.arange(1000, 9000), size=max(RECORD_COUNTS) - 1,
                         replace=False)
        color_ids = rng.permutation(np.arange(len(ids)) % len(COLORS))
        records = [(int(account), COLORS[color]) for account, color in zip(ids, color_ids)]
        for count in RECORD_COUNTS:
            # All lengths share the same final eight distractors.
            distractors = records[:count - 1 - SUFFIX_RECORDS] + records[-SUFFIX_RECORDS:]
            placements = (1, count // 2, count - 2, count - 1 - SUFFIX_RECORDS)
            for position, index in zip((*POSITIONS, "fixed suffix"), placements):
                for gold, color in enumerate(COLORS):
                    rows = distractors[:index] + [(key, color)] + distractors[index:]
                    yield report, key, count, position, index, gold, rows
            if count == 64:
                index = count // 2
                rows = distractors[:index] + [(9500 + report, COLORS[report % 2])] + distractors[index:]
                yield report, key, count, "absent", index, None, rows


def tokenized_register(tokenizer, key, index, gold, rows):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    row_texts = [f"\nAccount {account}: {color}" for account, color in rows]
    question = f"\nCode for account {key}:"
    text = HEADER + "".join(row_texts) + question
    header_ids = encode(HEADER)
    row_ids = [encode(row) for row in row_texts]
    question_ids = encode(question)
    input_ids = encode(text)
    if input_ids != header_ids + [token for row in row_ids for token in row] + question_ids:
        raise ValueError("token boundaries changed in the formatted prompt")
    if len({len(row) for row in row_ids}) != 1:
        raise ValueError("moving the evidence must preserve token length")
    context_tokens = sum(map(len, row_ids))
    question_start = len(header_ids) + context_tokens
    evidence_start = len(header_ids) + index * len(row_ids[0])
    evidence_end = evidence_start + len(row_ids[index])
    if sum(account == key for account, _ in rows) != int(gold is not None):
        raise ValueError("the queried key must occur exactly once when present")
    answer_ids = [encode(" " + color) for color in COLORS]
    if any(len(ids) != 1 for ids in answer_ids):
        raise ValueError("this probe requires one token for each candidate")
    if any(encode(text + " " + color) != input_ids + answer
           for color, answer in zip(COLORS, answer_ids)):
        raise ValueError("answer tokens must start after the prompt boundary")
    return input_ids, [ids[0] for ids in answer_ids], {
        "context_tokens": context_tokens,
        "input_tokens": len(input_ids),
        "question_start": question_start,
        "evidence_start": evidence_start if gold is not None else None,
        "evidence_end": evidence_end if gold is not None else None,
        "distance": question_start - evidence_end if gold is not None else None,
        "input_ids_sha256": hashlib.sha256(np.asarray(input_ids, dtype="<i4").tobytes()).hexdigest(),
    }


def run(cache_dir, offline):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama import modeling_llama

    from ..ch16_adaptation.real_adaptation import MODEL, REVISION, file_sha256

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    common = dict(revision=REVISION, cache_dir=str(cache_dir), local_files_only=offline,
                  trust_remote_code=False, token=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, **common)
    model, loading = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="sdpa", output_loading_info=True, **common)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"checkpoint did not load exactly: {loading}")
    model.eval().float().requires_grad_(False)
    checkpoint = hf_hub_download(MODEL, "model.safetensors", revision=REVISION,
                                 cache_dir=str(cache_dir), local_files_only=offline, token=False)
    observations = []
    with torch.inference_mode():
        for report, key, count, position, index, gold, rows in register_cases():
            input_ids, answers, details = tokenized_register(tokenizer, key, index, gold, rows)
            if len(input_ids) + 1 > model.config.max_position_embeddings:
                raise ValueError("prompt and one-token answer exceed the configured context")
            inputs = torch.tensor([input_ids])
            logits = model(inputs, use_cache=False, logits_to_keep=1).logits[0, -1].double()
            pair = logits[answers]
            log_prob = logits.log_softmax(dim=0)
            greedy = int(logits.argmax())
            observations.append({
                "report": report + 1, "key": key, "records": count,
                "placement": position, "record_index": index, "gold": gold,
                **details, "candidate_ids": answers, "candidate_logits": pair.tolist(),
                "candidate_mass": float(log_prob[answers].exp().sum()),
                "p_blue": float(pair.softmax(dim=0)[1]),
                "choice": int(pair.argmax()), "greedy_token": greedy,
                "greedy_text": tokenizer.decode([greedy]),
                "greedy_correct": greedy == answers[gold] if gold is not None else None,
            })
            if position == "fixed suffix" and count == RECORD_COUNTS[-1] and gold == 1:
                print(f"Measured report {report + 1}/{REPORTS}", flush=True)
    return {
        "model": MODEL, "revision": REVISION, "weights_sha256": file_sha256(checkpoint),
        "runtime_source_sha256": file_sha256(modeling_llama.__file__),
        "seed": SEED, "device": "cpu", "threads": 4, "dtype": "float32",
        "attention_backend": "sdpa", "header": HEADER, "colors": list(COLORS),
        "configured_context_tokens": model.config.max_position_embeddings,
        "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "torch", "transformers", "tokenizers", "safetensors")},
        "observations": observations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("output/model-cache"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cache_dir, args.offline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(result['observations'])} model evaluations to {args.output}")


if __name__ == "__main__":
    main()
