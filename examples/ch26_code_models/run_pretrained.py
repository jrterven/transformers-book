"""Generate and score the pinned local code-model runs in Chapter 26."""

import argparse
from contextlib import closing
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import sys

from .demo_tasks import (BASE, FIM_MARKERS, FIM_PREFIX, FIM_SUFFIX, GENERATE, OPERATOR_PREFIX,
                         evaluate, generation_cases, overlap_cases, repair_cases, repair_prompt)
from .model import pass_at_k


MODELS = {
    "fim15": {
        "repo": "mlx-community/Qwen2.5-Coder-1.5B-4bit",
        "revision": "a18a8bf7bf0563001cb807abff12a98d4a1ae43c",
        "weights": {
            "model.safetensors": "87869d23c53d1274641597bb96ab01471f2027c710b9fd09ca9b538a1da43ac1",
        },
    },
    "qwen38": {
        "repo": "mlx-community/Qwen3.8-27B-4bit",
        "revision": "b935172e4bb5dcd0543774a34748fb92422e3027",
        "weights": {
            "model-00001-of-00003.safetensors": "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d",
            "model-00002-of-00003.safetensors": "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670",
            "model-00003-of-00003.safetensors": "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a",
        },
    },
    "liquid24": {
        "repo": "LiquidAI/LFM2.5-24B-A2B-MLX-4bit",
        "revision": "667408d494bde191c2cb9dd4f4549827816b9275",
        "weights": {
            "model-00001-of-00003.safetensors": "17e953456c62314a498f1ef5ee8a5f1b51ac83f6b6bb684f160dde4f92cd11c0",
            "model-00002-of-00003.safetensors": "c3ae09dba31b3786c325425435021cf5d5bf06c1a79acb31a741e15a47e5bb14",
            "model-00003-of-00003.safetensors": "9814144af5045ea8c96e2f1352148fdbe9f8a0085b76df6d6760efbe916e9840",
        },
    },
    "gemma26": {
        "repo": "mlx-community/gemma-4-26B-A4B-it-qat-4bit",
        "revision": "0e3cbab38ce568cf6e23543010d08d03b731910c",
        "weights": {
            "model-00001-of-00003.safetensors": "898863f0d5eba7153a9c9b9443aac1969809f2149e88481c725b6da40783f129",
            "model-00002-of-00003.safetensors": "2abdca76ed4a647cfe8af680908894e2709b4e1221ddbbbef1b1edb14f60e303",
            "model-00003-of-00003.safetensors": "a554f0fdb61bc9e510a5318e723a780c31434fd3e3dcdd3c16c700f835ade7b7",
        },
    },
}
SEEDS = tuple(range(2601, 2609))
FENCE = re.compile(r"(?ms)^ *\x60{3}(?:python|py)?[ \t]*\n(.*?)^ *\x60{3}[ \t]*$")


def checkpoint_path(key, local_root, offline):
    from huggingface_hub import snapshot_download

    entry = MODELS[key]
    if local_root:
        path = local_root / key
    else:
        path = Path(snapshot_download(entry["repo"], revision=entry["revision"], local_files_only=offline,
                                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"]))
    for name, expected in entry["weights"].items():
        with (path / name).open("rb") as file:
            if file_digest(file, "sha256").hexdigest() != expected:
                raise ValueError(f"Unexpected weights for {key}/{name}")
    return path


def decode(model, tokenizer, prompt, *, temperature, top_p, seed, limit):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    pieces = []
    for step in stream_generate(model, tokenizer, ids, max_tokens=limit,
                                sampler=make_sampler(temp=temperature, top_p=top_p)):
        pieces.append(step.text)
    return {"raw": "".join(pieces), "prompt_tokens": len(ids), "prompt": prompt, "seed": seed,
            "temperature": temperature, "top_p": top_p, "top_k": 0,
            "repetition_penalty": 1.0, "max_tokens": limit,
            "generated_tokens_including_stop": step.generation_tokens, "finish": step.finish_reason}


def operator_probabilities(model, tokenizer, prompt):
    from mlx_lm import stream_generate

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    with closing(stream_generate(model, tokenizer, ids, max_tokens=1)) as steps:
        step = next(steps)
    probabilities = {}
    for operator in (" <", " <="):
        token, = tokenizer.encode(operator, add_special_tokens=False)
        probabilities[operator.strip()] = {"token_id": token, "probability": math.exp(step.logprobs[token].item())}
    return {"prompt": prompt, "prompt_tokens": len(ids), "probabilities": probabilities}


def fim_prompt(prefix, suffix):
    before, after, middle = FIM_MARKERS
    return f"{before}{prefix}{after}{suffix}{middle}"


def run_fim(model, tokenizer):
    conditions = []
    for name, suffix in (("half_open", FIM_SUFFIX), ("closed", FIM_SUFFIX.replace("is False", "is True", 1))):
        draw = decode(model, tokenizer, fim_prompt(FIM_PREFIX, suffix),
                      temperature=0, top_p=1, seed=2600, limit=64)
        source = FIM_PREFIX + draw["raw"]
        cases = overlap_cases()
        if name == "closed":
            for case in cases:
                _, (a, b), _ = case
                case[2] = bool(set(range(a[0], a[1]+1)) & set(range(b[0], b[1]+1)))
        if draw["finish"] == "stop":
            scored = evaluate(source, "overlaps", cases)
            scored["visible_suffix_checks"] = evaluate(source, "overlaps", [
                ["touching assertion", [[0, 2], [2, 5]], name == "closed"],
                ["one shared assertion", [[0, 3], [2, 5]], True],
            ])["checks"]
        else:
            scored = {"status": "limit", "passed": False, "checks": []}
        draw.update(condition=name, prefix=FIM_PREFIX, suffix=suffix, candidate=source, evaluation=scored)
        conditions.append(draw)
    fixed_prefix = FIM_PREFIX + OPERATOR_PREFIX
    traces = {
        name: operator_probabilities(model, tokenizer, fim_prompt(fixed_prefix, suffix))
        for name, suffix in (("no_suffix", ""), ("half_open", FIM_SUFFIX),
                             ("closed", FIM_SUFFIX.replace("is False", "is True", 1)))
    }
    return {"marker_ids": {token: tokenizer.encode(token, add_special_tokens=False) for token in FIM_MARKERS},
            "completions": conditions, "operator_traces": traces}


def extract_code(raw):
    # The final answer follows the native reasoning block when the model emits one.
    answer = raw.rsplit("</think>", 1)[-1].strip()
    if answer.startswith("\x60\x60\x60"):
        block = FENCE.fullmatch(answer)
        if block:
            return block[1].strip() + "\n"
    return answer + "\n"


def run_chat(key, model, tokenizer):
    def ask(text, function, cases, temperature, top_p, seed):
        options = {"enable_thinking": False} if key in {"qwen38", "gemma26"} else {}
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                tokenize=False, add_generation_prompt=True, **options)
        draw = decode(model, tokenizer, prompt, temperature=temperature, top_p=top_p, seed=seed,
                      limit=1024)
        if draw["finish"] == "stop":
            candidate = extract_code(draw["raw"])
            draw.update(candidate=candidate, evaluation=evaluate(candidate, function, cases))
        else:
            draw.update(candidate=None, evaluation={"status": "limit", "passed": False, "checks": []})
        return draw

    examples = generation_cases()
    samples = []
    for seed in SEEDS:
        print(f"{key}: generation seed {seed}", flush=True)
        samples.append(ask(GENERATE, "peak_bookings", examples, 0.8, 0.95, seed))
    correct = sum(sample["evaluation"]["passed"] for sample in samples)
    repairs = {}
    for condition, with_policy in (("issue_only", False), ("with_policy", True)):
        print(f"{key}: repair {condition}", flush=True)
        repairs[condition] = ask(repair_prompt(with_policy), "conflicts", repair_cases(), 0, 1, 2600)
    return {
        "generation": {"case_count": len(examples), "samples": samples, "correct": correct,
                       "pass_at_1": pass_at_k(len(samples), correct, 1),
                       "pass_at_2": pass_at_k(len(samples), correct, 2)},
        "repair": {"baseline": evaluate(BASE, "conflicts", repair_cases()), **repairs},
    }


def main(args):
    import mlx.core as mx
    from mlx_lm import load

    key = args.model
    model_path = checkpoint_path(key, args.model_root, args.offline)
    config_hashes = {}
    for path in sorted(model_path.iterdir()):
        if path.suffix in {".json", ".jinja"}:
            config_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    print(f"Loading {key} on {mx.default_device()}", flush=True)
    model, tokenizer = load(str(model_path), tokenizer_config={"trust_remote_code": False})
    report = {
        "model": key, "checkpoint": MODELS[key], "config_sha256": config_hashes,
        "runtime": {package: version(package) for package in ("mlx", "mlx-lm", "transformers", "tokenizers")},
        "python": sys.version.split()[0], "platform": platform.platform(),
        "results": run_fim(model, tokenizer) if key == "fim15" else run_chat(key, model, tokenizer),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--model-root", type=Path, help="use checkpoint subdirectories already on disk")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
