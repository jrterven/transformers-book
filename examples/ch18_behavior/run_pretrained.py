"""Patch a pinned Qwen model and evaluate routing policies on authored requests."""

import argparse
from contextlib import ExitStack
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path

import torch

from .data import PAIRS, SLICES, prompt
from .model import patch_effect, route_policy, summarize

MODEL = "Qwen/Qwen3.5-0.8B-Base"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
FILES = {
    "model.safetensors-00001-of-00001.safetensors":
        "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
}
SITE_NAMES = ("Before edit", "Edited verb", "After edit", "Answer", "Colon")
MIN_CONFIDENCE = 0.80


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


# snippet:replace_rows
def replace_rows(output, positions, donor_vectors):
    """In each batch row replace one token's whole block-output vector."""
    patched = output.clone()
    batch = torch.arange(output.shape[0], device=output.device)
    patched[batch, positions, :] = donor_vectors
    return patched
# end-snippet:replace_rows


def continuation_ids(tokenizer):
    candidates = [tokenizer.encode(" " + c, add_special_tokens=False) for c in "ABC"]
    if any(len(ids) != 1 for ids in candidates) or len(set(map(tuple, candidates))) != 3:
        raise ValueError("A, B and C must be distinct single-token continuations")
    return [ids[0] for ids in candidates]


def token_batch(tokenizer, requests, candidates):
    ids = []
    for request in requests:
        text = prompt(request)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        for label, token_id in zip("ABC", candidates, strict=True):
            joined = tokenizer.encode(text + " " + label, add_special_tokens=False)
            if joined != encoded + [token_id]:
                raise ValueError("candidate retokenizes the end of the prefix")
        ids.append(encoded)
    if len({len(row) for row in ids}) != 1:
        raise ValueError("patching requires paired inputs of equal token length")
    return torch.tensor(ids)


def last_logits(model, ids):
    return model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1, :]


def load_checkpoint(checkpoint):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    directory = checkpoint_directory(checkpoint)
    options = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(directory, **options)
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        directory, **options, output_loading_info=True, dtype=torch.float32,
        attn_implementation="eager",
    )
    if any(loading.values()):
        raise ValueError(f"the pinned checkpoint did not load exactly: {loading}")
    return model.eval(), tokenizer


def measure_pair(model, tokenizer, candidates, requests, block_indices):
    ids = token_batch(tokenizer, requests, candidates)
    changed = (ids[0] != ids[1]).nonzero().flatten().tolist()
    if len(changed) != 1:
        raise ValueError("a matched pair must differ at exactly one token")
    edit = changed[0]
    positions = torch.tensor([edit - 1, edit, edit + 1, ids.shape[1] - 2, ids.shape[1] - 1])
    if len(positions.unique()) != len(SITE_NAMES) or positions[0] < 0:
        raise ValueError("expected distinct in-range sites")
    blocks = model.model.language_model.layers
    observed = {}

    def record_block(index):
        def save_output(_, _inputs, output):
            observed[index] = output[:, positions, :].clone()
        return save_output

    with ExitStack() as cleanup:
        for index in block_indices:
            cleanup.callback(blocks[index].register_forward_hook(record_block(index)).remove)
        all_logits = last_logits(model, ids)
        native = all_logits[:, candidates]
    gaps = native[:, 1] - native[:, 0]

    # Replicate both recipient prompts, one per site, and retain two self-patch controls.
    n = len(SITE_NAMES)
    grid_positions = torch.cat((positions, positions, positions[-1:].repeat(2)))
    recipients = torch.cat((ids[0:1].repeat(n, 1), ids[1:2].repeat(n, 1), ids))
    patch_rows = []
    for index in block_indices:
        vectors = observed[index]
        donors = torch.cat((vectors[1], vectors[0], vectors[:, -1, :]), dim=0)

        def replace_output(_, _inputs, output):
            return replace_rows(output, grid_positions, donors)

        with blocks[index].register_forward_hook(replace_output):
            patched = last_logits(model, recipients)[:, candidates]
        torch.testing.assert_close(patched[-2:], native, atol=3e-4, rtol=0)
        # The site strictly before the edit has the same history in the two runs.
        torch.testing.assert_close(patched[[0, n]], native, atol=3e-4, rtol=0)
        # A last-block edit before the final position has no downstream token mixing.
        if index == len(blocks) - 1:
            unchanged = torch.cat((patched[:n-1], patched[n:2*n-1]))
            torch.testing.assert_close(unchanged, native.repeat_interleave(n-1, 0),
                                       atol=3e-4, rtol=0)
            torch.testing.assert_close(patched[[n-1, 2*n-1]], native.flip(0), atol=3e-4, rtol=0)
        values = patched[:, 1] - patched[:, 0]
        directions = []
        for recipient, donor in ((0, 1), (1, 0)):
            direction = []
            for row in range(recipient*n, (recipient+1)*n):
                gap = float(values[row])
                shift, recovery = patch_effect(float(gaps[recipient]), float(gaps[donor]), gap)
                direction.append({"logits": patched[row].tolist(), "gap": gap,
                                  "shift": shift, "recovery": recovery})
            directions.append(direction)
        patch_rows.append({
            "block": index + 1,
            "read_recipient": directions[0], "write_recipient": directions[1],
            "self_patch_logits": patched[-2:].tolist(),
            "self_patch_max_logit_difference": float((patched[-2:] - native).abs().max()),
        })
    return {
        "requests": requests, "token_count": ids.shape[1], "token_positions": (positions + 1).tolist(),
        "token_texts": [[tokenizer.decode(int(row[pos])) for pos in positions] for row in ids],
        "logits": native.tolist(), "gaps": gaps.tolist(), "patches": patch_rows,
        "top_tokens": [tokenizer.decode(int(token)) for token in all_logits.argmax(-1)],
    }


def evaluate_routes(model, tokenizer, candidates):
    outcomes = {}
    for group, cases in SLICES.items():
        rows = []
        for request, target in cases:
            ids = token_batch(tokenizer, [request], candidates)
            logits = last_logits(model, ids)[0].double()
            log_scores = logits[candidates]
            top = int(logits.argmax())
            predictions = {
                "Free first token": "ABC"[candidates.index(top)] if top in candidates else "invalid",
                "Constrained": route_policy(log_scores.tolist()),
                "Defer below 0.8": route_policy(log_scores.tolist(), MIN_CONFIDENCE),
            }
            rows.append({
                "request": request, "target": target, "top_token": tokenizer.decode(top),
                "top_token_id": top, "candidate_logits": log_scores.tolist(),
                "candidate_mass": float((log_scores.logsumexp(0) - logits.logsumexp(0)).exp()),
                "restricted_probabilities": log_scores.softmax(0).tolist(),
                "predictions": predictions,
            })
        outcomes[group] = {
            "cases": rows,
            "metrics": {policy: summarize([row["target"] for row in rows],
                                          [row["predictions"][policy] for row in rows])
                        for policy in rows[0]["predictions"]},
        }
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model, tokenizer = load_checkpoint(args.checkpoint)
    candidates = continuation_ids(tokenizer)
    layers = len(model.model.language_model.layers)
    with torch.inference_mode():
        routing = evaluate_routes(model, tokenizer, candidates)
        pairs = []
        for index, pair in enumerate(PAIRS):
            blocks = range(layers) if index == 0 else (0, 7, 15, 22, layers - 1)
            pairs.append(measure_pair(model, tokenizer, candidates, pair, blocks))
            print(f"Pair {index + 1}: gaps {pairs[-1]['gaps']}", flush=True)
    report = {
        "model": MODEL, "revision": REVISION, "sha256": FILES,
        "device": "cpu", "float": "float32", "threads": 4,
        "versions": {name: version(name) for name in ("torch", "transformers", "huggingface-hub", "safetensors")},
        "prompt_template": prompt("{request}"),
        "candidate_ids": dict(zip("ABC", candidates)),
        "site_names": SITE_NAMES, "min_confidence": MIN_CONFIDENCE, "pairs": pairs, "routing": routing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({group: data["metrics"] for group, data in routing.items()}, indent=2))


if __name__ == "__main__":
    main()
