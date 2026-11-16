"""Inspect a pinned Qwen hybrid and score answers to a paired parcel task."""

import argparse
from hashlib import file_digest
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path


MODEL = "Qwen/Qwen3.5-0.8B-Base"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
FILES = {
    "model.safetensors-00001-of-00001.safetensors":
        "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
}
MODES = ("full", "recent_only", "register_moved", "selected_record")
CODE_PAIRS = (("Cedar", "Maple"), ("Amber", "Coral"), ("North", "South"), ("Jade", "Pearl"))
LENGTHS = (32, 64, 128, 256, 512)
NOTICES = (
    "Notice: a crate was weighed.\n"
    "Notice: a receipt was printed.\n"
    "Notice: the window was open.\n"
    "Notice: the desk was swept.\n"
)


def make_case(codes, swapped, active):
    parcels = ("A", "B")
    assigned = codes[::-1] if swapped else codes
    records = {p: f"Register: parcel {p} has code {code}.\n"
               for p, code in zip(parcels, assigned, strict=True)}
    header = ("Parcel delivery log. The long-term register records each parcel code. "
              "Read the latest selection.\n\n")
    recent = NOTICES + f"Select parcel A for delivery.\nSelect parcel {active} for delivery.\n"
    question = (f"Possible codes: {codes[0]} or {codes[1]}.\n"
                "Question: What is the code of the active parcel?\nAnswer:")
    register = "".join(records.values())
    return {
        "codes": list(codes), "register": dict(zip(parcels, assigned, strict=True)),
        "active": active, "swapped": swapped, "answer": assigned[parcels.index(active)],
        "prompts": {
            "full": header + register + recent + question,
            "recent_only": header + recent + question,
            "register_moved": header + recent + register + question,
            "selected_record": header + recent + records[active] + question,
        },
    }


def checkpoint_directory(local):
    if local is None:
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(
            MODEL, revision=REVISION, token=False, allow_patterns=["*.json", "*.txt", "*.safetensors"]
        ))
    for name, expected in FILES.items():
        with (local / name).open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"file differs from the pinned checkpoint: {name}")
    return local


def cache_bytes(cache, torch):
    totals = dict(kv=0, state=0, conv=0)
    for layer in cache.layers:
        for name, category in (("keys", "kv"), ("values", "kv"),
                               ("recurrent_states", "state"), ("conv_states", "conv")):
            value = getattr(layer, name, None)
            if value is None:
                continue
            tensors = value.values() if isinstance(value, dict) else (value,)
            for tensor in tensors:
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"unexpected cache payload for {name}")
                totals[category] += tensor.numel() * tensor.element_size()
    return totals


def measure_cache(model, tokenizer, torch):
    text = "The parcel arrived after the desk opened. The driver read the delivery note. " * 50
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    records = []
    for length in LENGTHS:
        prefix = tokens[:, :length]
        if prefix.shape[1] != length:
            raise ValueError("the text must contain enough tokens for the cache measurement")
        result = model(prefix, use_cache=True, logits_to_keep=1)
        records.append({"tokens": length, **cache_bytes(result.past_key_values, torch)})
    return records


def score_cases(model, tokenizer):
    scores_by_prompt = {}
    cases = []
    for codes, swapped, active in product(CODE_PAIRS, (False, True), ("A", "B")):
        case = make_case(codes, swapped, active)
        encoded = [tokenizer.encode(" " + name, add_special_tokens=False) for name in codes]
        if any(len(ids) != 1 for ids in encoded) or encoded[0] == encoded[1]:
            raise ValueError(f"expected two distinct single-token completions: {codes}")
        candidates = [ids[0] for ids in encoded]
        case["candidate_ids"] = candidates
        case["scores"] = {}
        for mode, prompt in case.pop("prompts").items():
            if prompt not in scores_by_prompt:
                batch = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
                logits = model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1]
                chosen = logits[candidates].double()
                scores_by_prompt[prompt] = {
                    "input_tokens": batch.input_ids.shape[1],
                    "logits": chosen.tolist(),
                    "probabilities": chosen.softmax(dim=-1).tolist(),
                    "choice": codes[chosen.argmax().item()],
                    "candidate_probability_mass": float((chosen.logsumexp(0) - logits.double().logsumexp(0)).exp()),
                }
            case["scores"][mode] = scores_by_prompt[prompt]
        cases.append(case)
    return cases


def summarize(cases):
    def correct(case, mode):
        return case["scores"][mode]["choice"] == case["answer"]

    lookup = {(tuple(row["codes"]), row["swapped"], row["active"]): row for row in cases}
    results = {}
    for mode in MODES:
        registry_pairs = [
            (lookup[(codes, False, active)], lookup[(codes, True, active)])
            for codes, active in product(CODE_PAIRS, ("A", "B"))
        ]
        selection_pairs = [
            (lookup[(codes, swapped, "A")], lookup[(codes, swapped, "B")])
            for codes, swapped in product(CODE_PAIRS, (False, True))
        ]
        results[mode] = {
            "correct": sum(correct(row, mode) for row in cases),
            "total": len(cases),
            "registry_pairs": sum(correct(a, mode) and correct(b, mode) for a, b in registry_pairs),
            "registry_total": len(registry_pairs),
            "selection_pairs": sum(correct(a, mode) and correct(b, mode) for a, b in selection_pairs),
            "selection_total": len(selection_pairs),
        }
    return results


def main():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = checkpoint_directory(args.checkpoint)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, output_loading_info=True,
        dtype=torch.float32, attn_implementation="eager",
    )
    if any(loading.values()):
        raise ValueError(f"the pinned checkpoint did not load exactly: {loading}")
    model.eval()
    with torch.inference_mode():
        history = measure_cache(model, tokenizer, torch)
        cases = score_cases(model, tokenizer)
    report = {
        "model": MODEL, "revision": REVISION, "sha256": FILES, "device": "cpu", "dtype": "float32",
        "threads": 4, "versions": {name: version(name) for name in
                                   ("torch", "transformers", "huggingface-hub", "numpy", "safetensors")},
        "cache": history, "scores": summarize(cases), "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    print(json.dumps(report["scores"], indent=2))


if __name__ == "__main__":
    main()
