"""Inspect native Longformer windows, global relays, and retained mass.

All probes are fixed before reading model outputs. The optional experiment
does not change the regular NumPy examples or download a remote code module.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np


MODEL_ID = "allenai/longformer-base-4096"
REVISION = "301e6a42cb0d9976a6d6a26a079fef81c18aa895"
FILLER = "The report records routine dates, names, budgets, and project updates. "
TEXT = "The committee first met in Paris. " + FILLER * 45 + "In which city did the committee first meet?"
HEAD = 0


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def softmax64(scores):
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def expand_native(local, global_weights, n, radius, global_position):
    """Expand one native head for display, removing padding and filler rows."""
    dense = np.zeros((n, n), dtype=np.float64)
    globals_count = int(global_position is not None)
    for query in range(n):
        if query == global_position:
            # The local array's global-query row is filler, not its attention.
            assert np.count_nonzero(local[query]) == 0
            dense[query] = global_weights[:n, 0]
            assert np.count_nonzero(global_weights[n:]) == 0
            continue
        if globals_count:
            dense[query, global_position] = local[query, 0]
        for relative in range(-radius, radius + 1):
            key = query + relative
            weight = local[query, globals_count + radius + relative]
            if 0 <= key < n and key != global_position:
                dense[query, key] = weight
            else:
                assert weight == 0
    np.testing.assert_allclose(dense.sum(axis=1), 1, atol=1e-6, rtol=0)
    return dense


def observe(cache_dir, offline):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, LongformerForMaskedLM
    from .model import local_mask

    common = dict(revision=REVISION, cache_dir=str(cache_dir), token=False,
                  local_files_only=offline, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, **common)
    # This official checkpoint has no safetensors file. Restrict its pickle
    # loader to tensors; requirements enforce a current PyTorch release.
    model, loading = LongformerForMaskedLM.from_pretrained(
        MODEL_ID, **common, use_safetensors=False, weights_only=True,
        output_loading_info=True)
    assert not any(loading.get(k) for k in
                   ("missing_keys", "mismatched_keys", "error_msgs")), loading
    # The published file also includes a document pooler. The MLM encoder
    # intentionally does not instantiate or execute that two-tensor head.
    assert sorted(loading["unexpected_keys"]) == [
        "longformer.pooler.dense.bias", "longformer.pooler.dense.weight"], loading
    assert model.config._commit_hash == REVISION
    model.eval().float().requires_grad_(False)
    ids = tokenizer(TEXT, return_tensors="pt")["input_ids"]
    pieces = tokenizer.convert_ids_to_tokens(ids[0])
    n = ids.shape[1]
    source = pieces.index("ĠParis")
    target = max(i for i, piece in enumerate(pieces) if piece == "Ġcity")
    relay = n // 2
    replacement = tokenizer.encode(" Rome", add_special_tokens=False)
    assert len(replacement) == 1
    changed = ids.clone()
    changed[0, source] = replacement[0]
    windows = model.config.attention_window
    assert windows == [512] * 12
    radius = windows[0] // 2
    assert relay - source > radius and target - relay > radius
    assert target - source > 2 * radius
    assert n <= 4096
    first = model.longformer.encoder.layer[0].attention.self
    captured = {}
    arrays, report, states = {}, {}, {}

    def capture_input(module, args):
        captured["input"] = args[0][0, :n].detach().numpy().copy()

    def capture_output(module, args, result):
        captured["heads"] = result[0][0, :n].detach().numpy().copy()

    def project(x, layer):
        return (x.astype(np.float64) @ layer.weight.detach().numpy().astype(np.float64).T
                + layer.bias.detach().numpy().astype(np.float64))[:, :64]

    with torch.inference_mode():
        for name, global_position in (("local", None), ("global", relay)):
            global_mask = torch.zeros_like(ids)
            if global_position is not None:
                global_mask[0, global_position] = 1
            for variant, input_ids in (("base", ids), ("changed", changed)):
                handles = []
                if variant == "base":
                    handles = [first.register_forward_pre_hook(capture_input),
                               first.register_forward_hook(capture_output)]
                try:
                    result = model.longformer(
                        input_ids=input_ids, attention_mask=torch.ones_like(ids),
                        global_attention_mask=global_mask, output_hidden_states=True,
                        output_attentions=variant == "base", return_dict=True)
                finally:
                    for handle in handles:
                        handle.remove()
                # Copy complete first-two-block vectors, not just plot excerpts.
                states[name, variant] = np.stack([
                    h[0].detach().numpy().copy() for h in result.hidden_states[:3]])
                if variant == "base":
                    native_local = result.attentions[0][0, HEAD].detach().numpy().copy()
                    native_global = (None if global_position is None else
                        result.global_attentions[0][0, HEAD].detach().numpy().copy())
                    p = expand_native(native_local, native_global, n, radius, global_position)
                    allowed = local_mask(n, radius,
                        () if global_position is None else (global_position,))
                    assert np.count_nonzero(p[~allowed]) == 0
                    assert np.count_nonzero(p) == np.count_nonzero(allowed)
                    q, k, v = [project(captured["input"], layer)
                               for layer in (first.query, first.key, first.value)]
                    scores = q @ k.T / np.sqrt(64)
                    vg = None
                    if global_position is not None:
                        qg, kg, vg = [project(captured["input"], layer) for layer in
                                     (first.query_global, first.key_global, first.value_global)]
                        scores[relay] = qg[relay] @ kg.T / np.sqrt(64)
                        arrays["global_query_projection"] = qg
                        arrays["global_key_projection"] = kg
                        arrays["global_value_projection"] = vg
                    reconstructed = softmax64(np.where(allowed, scores, -np.inf))
                    expected_output = reconstructed @ v
                    if global_position is not None:
                        expected_output[relay] = reconstructed[relay] @ vg
                    native_output = captured["heads"][:, :64]
                    np.testing.assert_allclose(p, reconstructed, atol=2e-6, rtol=0)
                    np.testing.assert_allclose(expected_output, native_output, atol=2e-5, rtol=0)
                    arrays.update({name + "_" + key: value for key, value in {
                        "probabilities": p, "allowed": allowed, "scores": scores,
                        "q": q, "k": k, "v": v, "input": captured["input"],
                        "native_head_output": native_output,
                        "native_compact_attention": native_local,
                        "reconstructed_probabilities": reconstructed,
                    }.items()})
                    if native_global is not None:
                        arrays["native_global_attention"] = native_global
                    report[name] = {
                        "native_compact_head_shape": list(native_local.shape),
                        "native_global_head_shape": None if native_global is None else list(native_global.shape),
                        "allowed_pairs": int(allowed.sum()),
                        "max_abs_native_row_sum_error": float(np.max(np.abs(p.sum(axis=1) - 1))),
                        "probability_max_abs_reconstruction_error": float(np.max(np.abs(p - reconstructed))),
                        "head_output_max_abs_reconstruction_error": float(np.max(np.abs(expected_output - native_output))),
                        "target_source_probability": float(p[target, source]),
                        "target_relay_probability": float(p[target, relay]),
                        "relay_source_probability": float(p[relay, source]),
                    }
                del result
                print(name, variant, "complete", flush=True)

    differences = {}
    for name in ("local", "global"):
        delta = np.max(np.abs(states[name, "changed"] - states[name, "base"]), axis=-1)
        arrays[name + "_perturbation_all_positions"] = delta
        arrays[name + "_hidden_base"] = states[name, "base"]
        arrays[name + "_hidden_changed"] = states[name, "changed"]
        differences[name] = delta[:, [source, relay, target]].tolist()
    assert differences["local"][1][2] == differences["local"][2][2] == 0
    assert differences["global"][1][2] == 0
    assert differences["global"][1][1] > 0 and differences["global"][2][2] > 0
    # Mask shape changes kernel work, so unperturbed outputs across masks need
    # not agree; compare the two texts under each *same* mask instead.
    for name in ("local", "global"):
        assert differences[name][0][1] == differences[name][0][2] == 0

    # Post-hoc selection uses already-computed native weights: no DSA indexer.
    p = arrays["global_probabilities"][target]
    v = arrays["global_v"]
    legal = np.flatnonzero(arrays["global_allowed"][target])
    order = sorted(legal.tolist(), key=lambda i: (-p[i], i))
    ks = sorted(set([1, 2, 4, 8, 16, 32, 64, 128, 256, len(legal)]))
    ks = [k for k in ks if k <= len(legal)]
    # Normalize once in float64 to remove native float32 row-sum rounding.
    full_p = p / p.sum()
    full_output = full_p @ v
    vmax = np.linalg.norm(v[legal], axis=-1).max()
    retained, errors, bounds = [], [], []
    for count in ks:
        selected = np.array(order[:count])
        mass = full_p[selected].sum()
        weights = full_p[selected] / mass
        gathered = softmax64(arrays["global_scores"][target, selected][None])[0]
        np.testing.assert_allclose(weights, gathered, atol=2e-6, rtol=0)
        output = weights @ v[selected]
        error = float(np.linalg.norm(output - full_output))
        bound = float(2 * vmax * max(0, 1 - mass))
        assert error <= bound + 2e-12
        retained.append(float(mass))
        errors.append(error)
        bounds.append(bound)
    arrays.update(selection_k=np.array(ks), selection_mass=np.array(retained),
                  selection_error=np.array(errors), selection_bound=np.array(bounds))
    files = {}
    for filename in ("pytorch_model.bin", "config.json", "tokenizer.json", "vocab.json", "merges.txt"):
        file = hf_hub_download(MODEL_ID, filename, revision=REVISION,
                              cache_dir=str(cache_dir), local_files_only=True, token=False)
        files[filename] = {"bytes": Path(file).stat().st_size, "sha256": sha256(file)}
    report.update({
        "model": MODEL_ID, "revision": REVISION, "license": "Apache-2.0",
        "loading_info": loading, "loaded_parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_files": files, "text": TEXT, "filler_repetitions": 45,
        "input_ids": ids[0].tolist(), "token_pieces": pieces,
        "positions_zero_based": {"source": source, "relay": relay, "target": target},
        "changed_id": replacement[0], "changed_piece": tokenizer.convert_ids_to_tokens(replacement[0]),
        "probe_rules": {"source": "first Paris piece", "relay": "floor(token count / 2)",
                        "target": "last city piece", "head": HEAD, "blocks": [1, 2]},
        "config": {"layers": 12, "hidden_size": 768, "heads": 12, "head_width": 64,
                   "attention_window": windows, "local_radius": radius,
                   "input_positions": n, "padded_positions": int(np.ceil(n / 512) * 512)},
        "perturbation_max_abs_over_all_768_features": differences,
        "selection": {"query": target, "head": HEAD, "candidate_count": len(legal),
                      "k": ks, "retained_mass": retained, "head_output_l2_error": errors,
                      "twice_vmax_times_lost_mass": bounds, "vmax": float(vmax),
                      "method": "post-hoc top native attention weights, not an indexer or speedup"},
    })
    return arrays, report


def draw(arrays, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "figure.dpi": 160,
                         "savefig.dpi": 190, "axes.spines.top": False,
                         "axes.spines.right": False})
    n = report["config"]["input_positions"]
    s, g, t = [report["positions_zero_based"][k] for k in ("source", "relay", "target")]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")
    files = []

    def save(fig, name):
        fig.savefig(output / name, facecolor="white", bbox_inches="tight", pad_inches=.12)
        plt.close(fig)
        files.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, name, title in zip(axes, ("local", "global"),
                              ("(a) Published local window", "(b) One designated global position")):
        p = arrays[name + "_probabilities"]
        im = ax.imshow(np.ma.masked_where(p == 0, p), cmap=cmap,
                       norm=LogNorm(1e-6, 1), interpolation="nearest", rasterized=True)
        ax.set(title=title, xlabel="Key position (zero-based)", ylabel="Query position (zero-based)",
               xticks=[0, g, n-1], yticks=[0, g, n-1])
        ax.text(.03, .04, f"{report[name]['allowed_pairs']:,} allowed pairs",
                transform=ax.transAxes, color="black", fontsize=10,
                bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    fig.colorbar(im, ax=axes, label="Native attention probability (log scale)", shrink=.82)
    fig.suptitle("Longformer · block 1, head 1 · 648 actual input positions", fontsize=14)
    save(fig, "ch11_pretrained_windows.png")

    fig, axes = plt.subplots(2, 1, figsize=(10.7, 5.8), layout="constrained", sharex=True)
    axes[0].plot(arrays["local_probabilities"][t], label="Local only", color="#2563eb", lw=1.2)
    axes[0].plot(arrays["global_probabilities"][t], label="With global relay", color="#ea580c", lw=1)
    axes[0].set_title(f"(a) Ordinary query at city, position {t}", loc="left")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    axes[1].plot(arrays["global_probabilities"][g], color="#0f766e", lw=1)
    axes[1].set_title(f"(b) Global query at position {g}: separate learned global projections", loc="left")
    for ax in axes:
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set(ylim=(-1e-7, 1), ylabel="Attention probability", xlim=(-3, n+2))
        for x, label, color in ((s, "S: Paris", "#7c3aed"), (g, "G: relay", "#0f766e"), (t, "T: city", "#b45309")):
            ax.axvline(x, color=color, ls=":", lw=.9)
            ax.text(x, .38, label, rotation=90, ha="right", va="top", fontsize=9, color=color)
        ax.grid(axis="y", alpha=.18)
    axes[1].set_xlabel("Key position (zero-based); exact zeros sit on the baseline")
    fig.suptitle("Who can read the distant city in the first block?", fontsize=14)
    save(fig, "ch11_pretrained_relay_reads.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    matrices = [np.array(report["perturbation_max_abs_over_all_768_features"][k]).T
                for k in ("local", "global")]
    positive = np.concatenate([x[x > 0] for x in matrices])
    norm = LogNorm(float(positive.min() / 2), float(positive.max() * 2))
    for ax, values, title in zip(axes, matrices, ("(a) Local attention only", "(b) One global relay")):
        im = ax.imshow(np.ma.masked_equal(values, 0), cmap=cmap, norm=norm, aspect="auto")
        for y in range(3):
            for x in range(3):
                v = values[y, x]
                ax.text(x, y, "0" if v == 0 else f"{v:.2g}", ha="center", va="center",
                        fontsize=11, color="black", bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none", "pad": 1.5})
        ax.set(title=title, xticks=range(3), xticklabels=["Embedding", "After block 1", "After block 2"],
               yticks=range(3), yticklabels=[f"S: Paris, {s}", f"G: relay, {g}", f"T: city, {t}"])
    fig.colorbar(im, ax=axes, label="Max absolute feature change (log scale)", shrink=.8)
    fig.suptitle("Change only Paris → Rome; measure all 768 coordinates", fontsize=14)
    save(fig, "ch11_pretrained_perturbation.png")

    selection = report["selection"]
    ks = np.array(selection["k"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    axes[0].plot(ks, selection["retained_mass"], "o-", color="#2563eb")
    axes[0].set(ylabel="Retained probability mass ρ", ylim=(0, 1.04),
                title="(a) Keep the largest native weights")
    axes[1].plot(ks, selection["head_output_l2_error"], "o-", color="#ea580c", label="Measured head-output error")
    axes[1].plot(ks, selection["twice_vmax_times_lost_mass"], "s--", color="#0f766e", label="Bound: 2 Vmax (1 − ρ)")
    axes[1].set(ylabel="L2 distance over all 64 coordinates", title="(b) Renormalize selected values")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set(xlabel="Retained entries k (of 265 legal entries)", xticks=[1, 4, 16, 64, 265],
               xticklabels=[1, 4, 16, 64, 265])
        ax.grid(alpha=.2)
    fig.suptitle("Post-hoc diagnostic · block 1, head 1, query city · no indexer", fontsize=13)
    save(fig, "ch11_pretrained_retained_mass.png")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, report = observe(args.cache_dir, args.offline)
    files = draw(arrays, report, args.output_dir)
    np.savez_compressed(args.output_dir / "observations.npz", **arrays)
    report.update({"observed_on": "2026-11-09", "device": "cpu", "dtype": "float32",
                   "eval_mode": True, "weights_frozen": True, "threads": 1,
                   "sampling": False, "training": False,
                   "python": platform.python_version(), "platform": platform.platform(),
                   "software": {p: importlib.metadata.version(p) for p in
                                ("torch", "transformers", "tokenizers", "numpy", "matplotlib", "huggingface_hub")},
                   "source_sha256": sha256(__file__),
                   "outputs": {f: {"sha256": sha256(args.output_dir / f),
                                   "bytes": (args.output_dir / f).stat().st_size}
                               for f in files + ["observations.npz"]}})
    (args.output_dir / "PRETRAINED_FIGURES_RUN.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("local", "global", "positions_zero_based",
                    "perturbation_max_abs_over_all_768_features", "selection")}, indent=2))


if __name__ == "__main__":
    main()
