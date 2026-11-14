"""Measure several interpretation methods on the same pinned Qwen checkpoint."""

import argparse
from contextlib import ExitStack
from importlib.metadata import version
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import GUIDE, PAIRS, prompt
from .interpretability import fit_dictionary, integrated_gradients, ridge_probe
from .method_data import ATTRIBUTION_PAIR, TEST_FAMILIES, TRAIN_FAMILIES, probe_requests
from .run_pretrained import FILES, MODEL, REVISION, continuation_ids, last_logits, load_checkpoint

DEPTHS = (1, 4, 8, 12, 16, 20, 24)
HEAD_BLOCK = 16
STEERING_BLOCK = 16


def collect(model, tokenizer, cases, candidates):
    text_model = model.model.language_model
    blocks = text_model.layers
    # For the sparse dictionary keep the request field, including its label.
    # Verify the token boundaries before slicing off the guide and answer cue.
    guide_ids = tokenizer.encode(GUIDE, add_special_tokens=False)
    guide_length = len(guide_ids)
    answer_ids = tokenizer.encode("\nAnswer:", add_special_tokens=False)
    results = []
    for family, request, label in cases:
        cached = {}
        ids = tokenizer(prompt(request), add_special_tokens=False, return_tensors="pt").input_ids
        if ids[0, :guide_length].tolist() != guide_ids or ids[0, -len(answer_ids):].tolist() != answer_ids:
            raise ValueError("the guide or answer cue retokenizes at a field boundary")

        def save(depth):
            def hook(_, _args, output):
                cached[depth] = output[0].clone() if depth == STEERING_BLOCK else output[0, -1].clone()
            return hook

        with ExitStack() as cleanup:
            for depth in DEPTHS:
                cleanup.callback(blocks[depth-1].register_forward_hook(save(depth)).remove)
            logits = last_logits(model, ids)[0, candidates]

        residuals = torch.stack([cached[depth][-1] if depth == STEERING_BLOCK else cached[depth]
                                 for depth in DEPTHS])
        vocab_rows = model.lm_head.weight[candidates]
        lens = F.linear(text_model.norm(residuals), vocab_rows)
        torch.testing.assert_close(lens[-1], logits, rtol=0, atol=3e-4)
        results.append({"family": family, "request": request, "label": label, "ids": ids,
                        "residuals": residuals, "lens": lens, "logits": logits,
                        "request_vectors": cached[STEERING_BLOCK][guide_length:-len(answer_ids)].clone()})
    return results


def probe_scores(train, test):
    # The base chooses A or B. Probe models receive the same binary labels.
    y_train = torch.tensor([row["label"] for row in train], dtype=torch.float64)
    y_test = torch.tensor([row["label"] for row in test], dtype=torch.bool)
    all_train = torch.stack([row["residuals"] for row in train], 1).double()
    all_test = torch.stack([row["residuals"] for row in test], 1).double()
    generator = torch.Generator().manual_seed(18)
    controls = torch.stack([y_train[torch.randperm(len(y_train), generator=generator)]
                            for _ in range(24)], dim=1)
    results = []
    for depth, a, b in zip(DEPTHS, all_train, all_test, strict=True):
        # One fixed scale learned from all training coordinates, not labels.
        scale = (a - a.mean(0)).square().mean().sqrt() * a.shape[1] ** 0.5
        scores = ridge_probe(a/scale, y_train, b/scale)
        shuffled = ridge_probe(a/scale, controls, b/scale)
        correct = (scores >= 0.5) == y_test
        results.append({"block": depth, "correct": int(correct.sum()),
                        "shuffled_mean_correct": float(((shuffled >= 0.5) == y_test[:,None]).sum(0).double().mean()),
                        "scores": scores.tolist()})
    native = sum(bool(row["logits"][1] > row["logits"][0]) == row["label"] for row in test)
    return {"n": len(test), "native_correct": native, "layers": results}


def head_analysis(model, tokenizer, candidates):
    native_ids = tokenizer(prompt(PAIRS[0][0]), add_special_tokens=False, return_tensors="pt").input_ids
    block = model.model.language_model.layers[HEAD_BLOCK-1]
    attention = block.self_attn
    saved = {}

    def weights_hook(_, _args, output):
        saved["weights"] = output[1][0, :, -1].clone()

    def channels_hook(_, args):
        saved["channels"] = args[0][0, -1].clone()

    with attention.register_forward_hook(weights_hook), attention.o_proj.register_forward_pre_hook(channels_hook):
        native = last_logits(model, native_ids)[0,candidates]
    gap = float(native[1] - native[0])
    heads = saved["channels"].reshape(-1, attention.head_dim)
    # Gated per-head vectors are the actual input to W_O. This projection
    # isolates head write magnitudes before downstream nonlinear computation.
    output_weights = attention.o_proj.weight.reshape(attention.o_proj.out_features, len(heads), -1)
    writes = torch.einsum("dhk,hk->hd", output_weights, heads)
    torch.testing.assert_close(writes.sum(0), attention.o_proj(saved["channels"]), atol=1e-5, rtol=2e-5)
    ablated_gaps = []
    for head in range(len(heads)):
        def zero_head(_, args):
            amended = args[0].clone()
            start = head * attention.head_dim
            amended[:, -1, start:start + attention.head_dim] = 0
            return (amended,)

        with attention.o_proj.register_forward_pre_hook(zero_head):
            logits = last_logits(model, native_ids)[0,candidates]
        ablated_gaps.append(float(logits[1] - logits[0]))
    return {
        "block": HEAD_BLOCK, "native_gap": gap,
        "tokens": [tokenizer.decode(int(token)) for token in native_ids[0]],
        "attention_rows": saved["weights"].tolist(),
        "write_norms": writes.norm(dim=-1).tolist(),
        "ablation_gaps": ablated_gaps,
        "remove_effects": [gap-edited for edited in ablated_gaps],
    }


def steer(model, train, test, candidates):
    depth_index = DEPTHS.index(STEERING_BLOCK)
    examples = torch.stack([row["residuals"][depth_index] for row in train])
    labels = torch.tensor([row["label"] for row in train]).bool()
    direction = examples[labels].mean(0) - examples[~labels].mean(0)
    length = direction.norm()
    direction = direction / length
    random_direction = torch.randn(direction.shape, generator=torch.Generator().manual_seed(42))
    random_direction = random_direction / random_direction.norm()
    block = model.model.language_model.layers[STEERING_BLOCK-1]
    cases = [row for row in test if row["label"] == 0]
    outcomes = []
    for magnitude in (-2, -1, 0, 1, 2):
        row_scores = {}
        for name, vector in (("contrast", direction), ("random", random_direction)):
            scores = []
            for case in cases:
                def add_direction(_, _args, value):
                    output = value.clone()
                    output[:, -1] += magnitude * length * vector
                    return output

                with block.register_forward_hook(add_direction):
                    logits = last_logits(model, case["ids"])[0,candidates]
                scores.append(float(logits[1]-logits[0]))
            row_scores[name] = scores
        outcomes.append({"multiplier": magnitude, **row_scores})
    return {"block": STEERING_BLOCK, "n": len(cases), "direction_norm": float(length),
            "requests": [c["request"] for c in cases], "cases": outcomes}


def attribution(model, tokenizer, candidates, steps):
    ids = [tokenizer(prompt(request), add_special_tokens=False, return_tensors="pt").input_ids
           for request in ATTRIBUTION_PAIR]
    if ids[0].shape != ids[1].shape:
        raise ValueError("integrated gradients requires aligned embedding positions")
    text_model = model.model.language_model
    reference, target = [text_model.embed_tokens(row).detach() for row in ids]
    vocab_rows = model.lm_head.weight[candidates[:2]].detach()
    qwen_device = reference.device
    position_ids = torch.arange(reference.shape[1], device=qwen_device).unsqueeze(0)

    def score(embedding):
        vector = text_model(inputs_embeds=embedding, position_ids=position_ids,
                            use_cache=False).last_hidden_state[0,-1]
        a, b = F.linear(vector, vocab_rows)
        return b-a

    with torch.no_grad():
        base_gap, target_gap = float(score(reference)), float(score(target))
        for tokens, expected in zip(ids, (base_gap, target_gap), strict=True):
            native = last_logits(model, tokens)[0, candidates]
            if abs(expected - float(native[1] - native[0])) > 2e-4:
                raise ValueError("the differentiable score disagrees with the native endpoint")
    local_input = target.clone().requires_grad_(True)
    gradient, = torch.autograd.grad(score(local_input), local_input)
    local = ((target - reference) * gradient).sum(-1)[0].detach()
    integrated = integrated_gradients(score, reference, target, steps).sum(-1)[0]
    changed = (ids[0][0] != ids[1][0]).nonzero().flatten().tolist()
    return {
        "requests": ATTRIBUTION_PAIR, "steps": steps,
        "baseline_gap": base_gap, "target_gap": target_gap,
        "endpoint_change": target_gap-base_gap, "ig_sum": float(integrated.sum()),
        "local_sum": float(local.sum()),
        "positions": [index+1 for index in changed],
        "tokens": [[tokenizer.decode(int(x[0,index])) for index in changed] for x in ids],
        "ig": integrated[changed].tolist(), "local": local[changed].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ig-steps", type=int, default=24)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model, tokenizer = load_checkpoint(args.checkpoint)
    model.requires_grad_(False)
    candidates = continuation_ids(tokenizer)
    with torch.no_grad():
        print("Collecting frozen activations", flush=True)
        train = collect(model, tokenizer, probe_requests(TRAIN_FAMILIES, (31, 57, 93)), candidates)
        test = collect(model, tokenizer, probe_requests(TEST_FAMILIES, (12, 44, 85)), candidates)
        probes = probe_scores(train, test)
        print("Measuring attention writes and ablations", flush=True)
        heads = head_analysis(model, tokenizer, candidates)
        print("Testing contrastive and random directions", flush=True)
        steering = steer(model, train, test, candidates)
    print("Integrating embedding gradients", flush=True)
    gradients = attribution(model, tokenizer, candidates, args.ig_steps)
    print("Fitting sparse dictionaries", flush=True)
    train_tokens = torch.cat([row["request_vectors"] for row in train])
    test_tokens = torch.cat([row["request_vectors"] for row in test])
    dictionary = [fit_dictionary(train_tokens, test_tokens, sparsity=s) for s in (0.002, 0.01, 0.04)]
    record = {"model": MODEL, "revision": REVISION, "model_files": FILES, "threads": 4,
              "versions": {name: version(name) for name in ("torch","transformers","huggingface-hub","safetensors")},
              "tensor_dtype": "float32", "training_requests": len(train),
              "training_families": [family[0] for family in TRAIN_FAMILIES],
              "test_requests": [{"request": row["request"], "label": row["label"],
                                 "family": row["family"], "logits": row["logits"].tolist(),
                                 "lens": row["lens"].tolist()} for row in test],
              "probes": probes, "head": heads, "steering": steering, "attribution": gradients,
              "lens_depths": DEPTHS, "dictionary": dictionary, "dictionary_train_rows": len(train_tokens)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
