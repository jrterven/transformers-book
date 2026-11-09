"""Measure real decoder caches, grouped attention, and replay equivalence.

The fixed token IDs are replayed, never sampled. Ordinary chapter examples
remain independent of these optional pretrained-model dependencies.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np


TEXT = "The red car stopped near the bridge."
MODELS = {
    "smol": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
}
PREFILL = 2
PROBE = 2
FEATURES = tuple(range(8))
CHUNKS = (2, 1, 2, 3)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def softmax64(values):
    values = np.asarray(values, dtype=np.float64)
    exp = np.exp(values - values.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def snapshot(cache):
    """Copy arrays before the mutable DynamicCache receives another append."""
    return np.stack([np.stack([x.detach().numpy().copy() for x in pair]) for pair in cache])


def payload(cache):
    return sum(x.numel() * x.element_size() for pair in cache for x in pair)


def observe(name, cache_dir, offline):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, revision = MODELS[name]
    common = dict(revision=revision, cache_dir=str(cache_dir), token=False,
                  local_files_only=offline, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **common)
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_id, **common, use_safetensors=True, output_loading_info=True,
        attn_implementation="eager")
    assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    assert model.config._commit_hash == revision
    model.eval().float().requires_grad_(False)
    ids = tokenizer(TEXT, return_tensors="pt", add_special_tokens=False)["input_ids"]
    assert ids.shape == (1, 8)
    pieces = [p.replace("Ġ", "·") for p in tokenizer.convert_ids_to_tokens(ids[0])]
    assert pieces[PROBE] == "·car"
    layers = model.config.num_hidden_layers
    heads = model.config.num_attention_heads
    kv_heads = getattr(model.config, "num_key_value_heads", heads)
    width = model.config.hidden_size // heads
    count = ids.shape[1]
    captured = {}

    def call(start, stop, cache=None, position_override=None):
        positions = torch.arange(start, stop)
        return model(input_ids=ids[:, start:stop], past_key_values=cache,
                     attention_mask=torch.ones((1, stop), dtype=torch.long),
                     position_ids=(positions if position_override is None else position_override)[None],
                     cache_position=positions, use_cache=True, output_attentions=True,
                     return_dict=True)

    def run(chunks, capture_probe=False):
        cache, start, logits, stages, sizes = None, 0, [], [], []
        for size in chunks:
            handles = []
            if capture_probe and start == PROBE:
                attention = model.model.layers[0].self_attn

                def before_attention(module, args, kwargs):
                    captured["input"] = kwargs["hidden_states"].detach().numpy().copy()
                    for key, value in zip(("cos", "sin"), kwargs["position_embeddings"]):
                        captured[key] = value.detach().numpy().copy()

                def before_output(module, args):
                    captured["head_outputs"] = args[0].detach().numpy().copy()

                handles = [attention.register_forward_pre_hook(before_attention, with_kwargs=True),
                           attention.o_proj.register_forward_pre_hook(before_output)]
            try:
                result = call(start, start + size, cache)
            finally:
                for handle in handles:
                    handle.remove()
            cache = result.past_key_values
            current = snapshot(cache)
            if stages:
                old_length = stages[-1].shape[-2]
                np.testing.assert_array_equal(current[..., :old_length, :], stages[-1])
            assert current.shape == (layers, 2, 1, kv_heads, start + size, width)
            assert current.dtype == np.float32
            actual_bytes = payload(cache)
            assert actual_bytes == layers * (start + size) * kv_heads * 2 * width * 4
            logits.append(result.logits.detach().numpy())
            stages.append(current)
            sizes.append({"length": start + size, "bytes": actual_bytes})
            if capture_probe and start == PROBE:
                captured["probabilities"] = result.attentions[0].detach().numpy().copy()
                captured["cache"] = current[0].copy()
            start += size
        assert start == count
        return np.concatenate(logits, axis=1), stages, sizes

    with torch.inference_mode():
        full = call(0, count)
        full_logits = full.logits.detach().numpy().copy()
        full_cache = snapshot(full.past_key_values)
        single, stages, sizes = run((PREFILL,) + (1,) * (count - PREFILL), name == "smol")
        chunked, chunk_stages, _ = run(CHUNKS)
    np.testing.assert_allclose(single, full_logits, atol=5e-4, rtol=0)
    np.testing.assert_allclose(chunked, full_logits, atol=5e-4, rtol=0)
    np.testing.assert_allclose(stages[-1], full_cache, atol=2e-4, rtol=2e-5)
    np.testing.assert_allclose(chunk_stages[-1], full_cache, atol=2e-4, rtol=2e-5)
    np.testing.assert_array_equal(single.argmax(-1), full_logits.argmax(-1))
    np.testing.assert_array_equal(chunked.argmax(-1), full_logits.argmax(-1))
    for block in full.attentions:
        matrix = block.detach().numpy()
        assert np.count_nonzero(np.triu(matrix, 1)) == 0
    arrays = {"ids": ids.numpy(), "full_logits": full_logits, "single_logits": single,
              "chunk_logits": chunked, "full_cache": full_cache}
    arrays.update({f"cache_after_{size['length']}": value for size, value in zip(sizes, stages)})
    errors_single = np.max(np.abs(full_logits - single), axis=(0, 2))
    errors_chunk = np.max(np.abs(full_logits - chunked), axis=(0, 2))
    report = {"model": model_id, "revision": revision, "loading": loading,
              "parameters": sum(p.numel() for p in model.parameters()), "pieces": pieces,
              "input_ids": ids[0].tolist(), "layers": layers, "query_heads": heads,
              "kv_heads": kv_heads, "head_width": width, "cache_dtype": "float32",
              "bytes_per_scalar": 4, "stages": sizes, "chunk_schedule": list(CHUNKS),
              "single_logit_max_errors_by_position": errors_single.tolist(),
              "chunk_logit_max_errors_by_position": errors_chunk.tolist(),
              "single_cache_max_error": float(np.max(np.abs(full_cache - stages[-1]))),
              "chunk_cache_max_error": float(np.max(np.abs(full_cache - chunk_stages[-1]))),
              "old_cache_entries_preserved_exactly": True, "formula_matches_every_payload": True,
              "all_argmax_predictions_match": True, "all_future_attention_entries_zero": True}
    if name == "smol":
        attention = model.model.layers[0].self_attn
        # Reconstruct all nine queries in float64 from the full 576-feature input.
        q = captured["input"].astype(np.float64) @ attention.q_proj.weight.numpy().astype(np.float64).T
        q = q.reshape(1, 1, heads, width).transpose(0, 2, 1, 3)
        rotate_half = np.concatenate((-q[..., width // 2:], q[..., :width // 2]), axis=-1)
        rotated = q * captured["cos"][:, None] + rotate_half * captured["sin"][:, None]
        keys, values = captured["cache"][0], captured["cache"][1]
        group = heads // kv_heads
        k_repeated, v_repeated = np.repeat(keys, group, axis=1), np.repeat(values, group, axis=1)
        scores = rotated @ k_repeated.swapaxes(-1, -2) / np.sqrt(width)
        probabilities = softmax64(scores)
        np.testing.assert_allclose(probabilities, captured["probabilities"], atol=3e-6, rtol=2e-5)
        head_outputs = probabilities @ v_repeated
        native_heads = captured["head_outputs"].reshape(1, 1, heads, width).transpose(0, 2, 1, 3)
        np.testing.assert_allclose(head_outputs, native_heads, atol=4e-6, rtol=3e-5)
        contributions = captured["probabilities"][0, 0, 0] * values[0, 0, :, 0]
        # Preserve correct cache slots while deliberately resetting only RoPE positions.
        with torch.inference_mode():
            prefix = call(0, PREFILL)
            wrong = call(PROBE, PROBE + 1, prefix.past_key_values, torch.zeros(1, dtype=torch.long))
        wrong_logits = wrong.logits[0, 0].numpy()
        correct_probs = softmax64(full_logits[0, PROBE])
        wrong_probs = softmax64(wrong_logits)
        top = np.argsort(-correct_probs, kind="stable")[:5]
        arrays.update(probe_scores=scores, probe_probabilities=captured["probabilities"],
                      probe_cache=captured["cache"], probe_head_outputs=native_heads,
                      probe_query_rotated=rotated, probe_value_contributions=contributions,
                      wrong_position_logits=wrong_logits, top_ids=top,
                      correct_top_probabilities=correct_probs[top], wrong_top_probabilities=wrong_probs[top])
        report["probe"] = {"position": PROBE, "block": 1, "query_head_indices": [0, 1, 2], "shared_kv_head_index": 0,
            "first_query_scores": scores[0, 0, 0].tolist(),
            "three_query_probability_rows": captured["probabilities"][0, :3, 0].tolist(),
            "first_value_coordinate": values[0, 0, :, 0].tolist(),
            "weighted_value_contributions": contributions.tolist(),
            "first_head_output_coordinate": float(native_heads[0, 0, 0, 0]),
            "probability_reconstruction_max_error": float(np.max(np.abs(probabilities - captured["probabilities"]))),
            "head_output_reconstruction_max_error": float(np.max(np.abs(head_outputs - native_heads))),
            "wrong_position_max_logit_change": float(np.max(np.abs(wrong_logits - single[0, PROBE]))),
            "correct_top_five_ids": top.tolist(),
            "correct_top_five_pieces": [repr(tokenizer.decode([int(i)])) for i in top],
            "correct_top_five_probabilities": correct_probs[top].tolist(),
            "wrong_position_same_five_probabilities": wrong_probs[top].tolist(),
            "top_five_remaining_probability": float(1 - correct_probs[top].sum())}
    weight_file = hf_hub_download(model_id, "model.safetensors", revision=revision,
                                  cache_dir=str(cache_dir), local_files_only=True, token=False)
    report["weights_sha256"] = sha256(weight_file)
    del model
    return report, arrays


def figures(reports, arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    paths = []

    def save(fig, name):
        path = output / name
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.12, facecolor="white")
        plt.close(fig)
        paths.append(path)

    smol, a = reports["smol"], arrays["smol"]
    lengths = (2, 3, 8)
    fig, axes = plt.subplots(2, 3, figsize=(10.3, 6.8), layout="constrained")
    for field, label in enumerate(("Rotated keys K", "Values V")):
        scale = float(np.max(np.abs(a["cache_after_8"][0, field, 0, 0, :, :8])))
        for column, length in enumerate(lengths):
            data = np.full((8, 8), np.nan)
            data[:length] = a[f"cache_after_{length}"][0, field, 0, 0, :, :8]
            ax = axes[field, column]
            im = ax.imshow(data, cmap="PuOr_r", vmin=-scale, vmax=scale, aspect="auto")
            ax.set_facecolor("#e5e7eb")
            ax.set_xticks(range(8), FEATURES, fontsize=8)
            ax.set_yticks(range(8), [f"{i}: {p}" for i, p in enumerate(smol["pieces"])] if column == 0 else range(8), fontsize=8)
            phase = "Prefill" if length == 2 else "After car" if length == 3 else "Complete input"
            ax.set_title((phase + "\n" if field == 0 else "") + f"{label}: {length} positions", fontsize=10)
            ax.set_xlabel("Feature array index")
            ax.add_patch(Rectangle((-.5, -.5), 8, 2, fill=False, edgecolor="#178473", linewidth=2))
        fig.colorbar(im, ax=axes[field, :], shrink=.7, label=f"{label} coordinate")
    fig.suptitle("SmolLM2, block 1, stored KV head 1: old rows are preserved", fontsize=12)
    save(fig, "ch10_pretrained_cache.png")

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.5), layout="constrained")
    p = a["probe_probabilities"][0, :3, 0]
    im = axes[0].imshow(p, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(3), ["0: The", "1: ·red", "2: ·car"])
    axes[0].set_yticks(range(3), ["Query head 1", "Query head 2", "Query head 3"])
    axes[0].set_title("(a) Three real queries at car read the same stored KV head", loc="left", fontsize=11)
    axes[0].axvline(1.5, color="#cf8a39", linestyle="--")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{p[i, j]:.6f}", ha="center", va="center", color="white" if p[i, j] > .55 else "#222222")
    fig.colorbar(im, ax=axes[0], label="Attention probability", shrink=.8)
    contributions = a["probe_value_contributions"]
    bars = axes[1].bar(range(3), contributions, color="#2985ad")
    axes[1].bar_label(bars, fmt="%.5f", padding=4, fontsize=9)
    total = smol["probe"]["first_head_output_coordinate"]
    low, high = min(contributions.min(), 0), max(contributions.max(), 0)
    span = high - low
    axes[1].set_ylim(low - .2 * span, high + .25 * span)
    axes[1].axhline(0, color="#666666", linewidth=.7)
    axes[1].set_xticks(range(3), ["The", "·red", "·car"])
    axes[1].set_ylabel("Weighted value coordinate 0")
    axes[1].set_title(f"(b) Query head 1: three contributions sum to {total:.6f}", loc="left", fontsize=11)
    save(fig, "ch10_pretrained_shared_reads.png")

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.0), layout="constrained")
    for name, color, marker in (("smol", "#178473", "o"), ("gpt2", "#64748b", "s")):
        r = reports[name]
        for method, style in (("single", "-"), ("chunk", "--")):
            axes[0].plot(range(8), r[f"{method}_logit_max_errors_by_position"], style, marker=marker,
                         color=color, label=f"{'SmolLM2' if name == 'smol' else 'GPT-2'}, {method}", markersize=4)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_xticks(range(8))
    axes[0].set_xlabel("Input array position")
    axes[0].set_ylabel("Max absolute vocabulary-logit error")
    axes[0].set_title("(a) Same fixed IDs: full pass versus cached replay", loc="left", fontsize=11)
    axes[0].legend(fontsize=8, ncol=2)
    x = np.arange(5)
    axes[1].bar(x - .18, a["correct_top_probabilities"], .36, label="Correct absolute positions", color="#178473")
    axes[1].bar(x + .18, a["wrong_top_probabilities"], .36, label="Reset only new RoPE position to zero", color="#cf8a39")
    axes[1].set_xticks(x, smol["probe"]["correct_top_five_pieces"], fontsize=8)
    axes[1].set_ylabel("Next-token probability")
    axes[1].set_title("(b) SmolLM2 after car: five most probable IDs under correct replay", loc="left", fontsize=10)
    axes[1].legend(fontsize=8)
    save(fig, "ch10_pretrained_replay.png")

    fig, ax = plt.subplots(figsize=(8.4, 4.2), layout="constrained")
    for name, color, marker in (("smol", "#178473", "o"), ("gpt2", "#2985ad", "s")):
        r = reports[name]
        lengths = np.array([s["length"] for s in r["stages"]])
        measured = np.array([s["bytes"] for s in r["stages"]]) / 1024
        formula = lengths * r["layers"] * r["kv_heads"] * 2 * r["head_width"] * 4 / 1024
        title = "SmolLM2: 30 layers, 3 stored KV heads" if name == "smol" else "GPT-2: 12 layers, 12 stored KV heads"
        ax.plot(lengths, formula, color=color, label=title)
        ax.scatter(lengths, measured, color=color, marker=marker, s=35)
        ax.annotate(f"{measured[-1]:.0f} KiB", (lengths[-1], measured[-1]), xytext=(8, 0), textcoords="offset points", color=color)
    ax.set_xlim(1.7, 9.2)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(2, 9))
    ax.set_xlabel("Stored token positions per sequence")
    ax.set_ylabel("K + V logical tensor payload, all layers (KiB)")
    ax.set_title("Real float32 caches: measured points match the shape formula", loc="left", fontsize=12)
    ax.legend(fontsize=9)
    ax.text(.02, .05, "Batch size 1; both use width 64 per key/value head\nDifferent checkpoints: this is storage accounting, not a speed or quality comparison",
            transform=ax.transAxes, fontsize=8)
    save(fig, "ch10_pretrained_bytes.png")
    return {path.name: sha256(path) for path in paths}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch10-pretrained"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reports, arrays = {}, {}
    for name in MODELS:
        reports[name], arrays[name] = observe(name, args.cache_dir, args.offline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "observations.npz"
    np.savez_compressed(archive, **{f"{name}_{key}": value for name, observation in arrays.items() for key, value in observation.items()})
    record = {"text": TEXT, "selection": "Fixed prompt; prefill 2; block 1; car at index 2; first KV group; first eight features; top-five IDs by a fixed rank rule",
              "execution": "Frozen CPU float32, eager attention, one thread, deterministic algorithms; fixed IDs replayed without sampling",
              "models": reports, "software": {"python": platform.python_version(), **{n: importlib.metadata.version(n) for n in
              ("torch", "transformers", "tokenizers", "huggingface_hub", "safetensors", "numpy", "matplotlib", "Pillow")}},
              "artifacts": {archive.name: sha256(archive), **figures(reports, arrays, args.output_dir)},
              "source_sha256": sha256(__file__)}
    (args.output_dir / "PRETRAINED_FIGURES_RUN.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
