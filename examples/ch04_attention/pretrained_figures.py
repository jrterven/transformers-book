"""Open a real attention head in a pinned, frozen DistilGPT2 checkpoint.

The default NumPy demonstration remains independent of this optional model.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from examples.ch01_language.pretrained_figures import MODEL, REVISION, readable, sha256
from .attention import scaled_dot_product_attention


TEXTS = ("the red car", "the red bike")
LAYER = 0  # First block, chosen before inspecting the values.
HEAD = 0   # First head, not selected for a linguistic pattern.
COORDINATES = tuple(range(8))


def observe(cache_dir, offline):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, GPT2LMHeadModel

    torch.set_num_threads(1)
    common = dict(revision=REVISION, cache_dir=str(cache_dir),
                  local_files_only=offline, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, **common)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL, **common, use_safetensors=True, attn_implementation="eager"
    ).eval().float().requires_grad_(False)
    assert model.config._commit_hash == REVISION
    weight_file = hf_hub_download(MODEL, "model.safetensors", revision=REVISION,
                                 cache_dir=str(cache_dir), local_files_only=True)
    attention = model.transformer.h[LAYER].attn
    assert attention.scale_attn_weights
    assert not attention.scale_attn_by_inverse_layer_idx
    assert not attention.reorder_and_upcast_attn
    captured = {}

    def capture_projection(module, inputs, output):
        captured["projection_input"] = inputs[0].detach().clone()
        captured["joint_qkv"] = output.detach().clone()

    def capture_read(module, inputs):
        captured["joined_head_reads"] = inputs[0].detach().clone()

    handles = [attention.c_attn.register_forward_hook(capture_projection),
               attention.c_proj.register_forward_pre_hook(capture_read)]
    traces, records = [], []
    try:
        with torch.inference_mode():
            for text in TEXTS:
                encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
                assert tokenizer.decode(encoded["input_ids"][0]) == text
                result = model(**encoded, output_attentions=True,
                               output_hidden_states=True, use_cache=False)
                joint = captured["joint_qkv"]
                projected64 = (captured["projection_input"][0].numpy().astype(np.float64)
                               @ attention.c_attn.weight.numpy().astype(np.float64)
                               + attention.c_attn.bias.numpy().astype(np.float64))
                np.testing.assert_allclose(projected64, joint[0].numpy(),
                                           atol=1e-5, rtol=3e-6)
                blocks = joint.split(attention.split_size, dim=-1)
                q, k, v = [t.reshape(1, t.shape[1], attention.num_heads,
                                    attention.head_dim).transpose(1, 2)[0]
                           for t in blocks]
                native_weights = result.attentions[LAYER][0]
                native_reads = captured["joined_head_reads"].reshape(
                    1, q.shape[1], attention.num_heads, attention.head_dim
                ).transpose(1, 2)[0]
                allowed = np.tril(np.ones((q.shape[1], k.shape[1]), dtype=bool))
                # Independent float64 NumPy core, using the observed float32 Q/K/V.
                reconstructed, weights64 = scaled_dot_product_attention(
                    q.numpy(), k.numpy(), v.numpy(), allowed)
                np.testing.assert_allclose(weights64, native_weights.numpy(),
                                           atol=2e-6, rtol=2e-6)
                np.testing.assert_allclose(reconstructed, native_reads.numpy(),
                                           atol=2e-5, rtol=3e-6)
                np.testing.assert_allclose(native_weights.numpy().sum(-1), 1, atol=2e-7, rtol=0)
                assert np.count_nonzero(native_weights.numpy()[:, ~allowed]) == 0
                values = {
                    "input_ids": encoded["input_ids"][0].numpy().copy(),
                    "projection_input": captured["projection_input"][0].numpy().copy(),
                    "joint_qkv": joint[0].numpy().copy(),
                    "reconstructed_joint_qkv": projected64,
                    "q": q.numpy().copy(), "k": k.numpy().copy(), "v": v.numpy().copy(),
                    "attention": native_weights.numpy().copy(),
                    "head_reads": native_reads.numpy().copy(),
                    "reconstructed_attention": weights64,
                    "reconstructed_head_reads": reconstructed,
                    "final_hidden": result.hidden_states[-1][0].numpy().copy(),
                    "all_hidden_states": np.stack([h[0].numpy() for h in result.hidden_states]),
                }
                traces.append(values)
                records.append({"text": text, "ids": values["input_ids"].tolist(),
                                "pieces": [readable(tokenizer, i) for i in values["input_ids"]]})
    finally:
        for handle in handles:
            handle.remove()

    first, changed = traces
    assert len(first["input_ids"]) == 3 == len(changed["input_ids"])
    np.testing.assert_array_equal(first["input_ids"][:2], changed["input_ids"][:2])
    assert first["input_ids"][2] != changed["input_ids"][2]
    for key in ("projection_input", "final_hidden"):
        np.testing.assert_allclose(first[key][:2], changed[key][:2], atol=1e-6, rtol=0)
    for key in ("q", "k", "v", "head_reads"):
        np.testing.assert_allclose(first[key][:, :2], changed[key][:, :2], atol=1e-6, rtol=0)
    np.testing.assert_allclose(first["all_hidden_states"][:, :2],
                               changed["all_hidden_states"][:, :2], atol=1e-6, rtol=0)
    q, k, v = [first[key][HEAD].astype(np.float64) for key in ("q", "k", "v")]
    raw_scores = q @ k.T
    scaled_scores = raw_scores / np.sqrt(q.shape[-1])
    weights = first["attention"][HEAD].astype(np.float64)
    contributions = weights[-1, :, None] * v
    np.testing.assert_allclose(contributions.sum(0), first["head_reads"][HEAD, -1],
                               atol=2e-5, rtol=3e-6)
    # Independently expand a dot product as 64 coordinate products.
    np.testing.assert_allclose((q[-1] * k).sum(-1), raw_scores[-1], atol=1e-12)
    read = first["head_reads"][HEAD]
    assert np.all(read >= v.min(0) - 2e-5) and np.all(read <= v.max(0) + 2e-5)
    arrays = {f"{prefix}_{key}": value for prefix, trace in zip(("original", "changed"), traces)
              for key, value in trace.items()}
    arrays.update(raw_scores=raw_scores, scaled_scores=scaled_scores,
                  last_query_contributions=contributions,
                  head_read_delta=changed["head_reads"][HEAD] - read)
    report = {
        "model": MODEL, "revision": REVISION, "weights_sha256": sha256(weight_file),
        "mode": "eval; frozen CPU float32 model; eager attention; no sampling or training",
        "reconstruction": "independent float64 NumPy attention on observed float32 Q/K/V",
        "layer": LAYER + 1, "head": HEAD + 1, "selection": "first block and first head, chosen before inspection",
        "layers": model.config.n_layer, "heads": attention.num_heads,
        "model_width": model.config.n_embd, "head_width": attention.head_dim,
        "scale_divisor": float(np.sqrt(attention.head_dim)),
        "projection": "layer-normalized block input, then affine Q/K/V projections with learned biases",
        "tokens": records, "displayed_coordinates": [i+1 for i in COORDINATES],
        "last_query": {"position": 3, "raw_scores": raw_scores[-1].tolist(),
                       "scaled_scores": scaled_scores[-1].tolist(),
                       "weights": weights[-1].tolist(),
                       "value_first_two": v[:, :2].tolist(),
                       "contributions_first_two": contributions[:, :2].tolist(),
                       "read_first_two": read[-1, :2].tolist()},
        "checks": {
            "maximum_projection_reconstruction_error": float(np.abs(first["reconstructed_joint_qkv"] - first["joint_qkv"]).max()),
            "maximum_attention_reconstruction_error": float(np.abs(first["reconstructed_attention"] - first["attention"]).max()),
            "maximum_read_reconstruction_error": float(np.abs(first["reconstructed_head_reads"] - first["head_reads"]).max()),
            "maximum_row_sum_error": float(np.abs(first["attention"].sum(-1) - 1).max()),
            "future_weight": float(np.abs(np.triu(first["attention"], k=1)).max()),
            "prefix_hidden_difference_all_layers": float(np.abs(first["all_hidden_states"][:, :2] - changed["all_hidden_states"][:, :2]).max()),
            "read_max_difference_by_position": np.abs(arrays["head_read_delta"]).max(-1).tolist(),
            "final_hidden_max_difference_by_position": np.abs(changed["final_hidden"] - first["final_hidden"]).max(-1).tolist(),
        },
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "tokenizers", "huggingface_hub",
                      "safetensors", "numpy", "matplotlib", "Pillow")},
    }
    return report, arrays


def draw_figures(report, arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    labels = report["tokens"][0]["pieces"]
    blue, orange = "#2468a0", "#c8772f"
    paths = []

    def heatmap(ax, values, signed, title, row_labels, column_labels, limit=None, decimals=2):
        limit = float(np.abs(values).max()) if limit is None else limit
        im = ax.imshow(values, cmap="PuOr_r" if signed else "Blues",
                       vmin=-limit if signed else 0, vmax=limit, aspect="auto")
        ax.set_xticks(range(values.shape[1]), column_labels)
        ax.set_yticks(range(values.shape[0]), row_labels)
        ax.set_title(title, loc="left", pad=10)
        for (r, c), value in np.ndenumerate(values):
            ax.text(c, r, f"{value:.{decimals}f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > .65*limit else "#222222")
        return im

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 6.3), layout="constrained")
    all_values = [arrays["original_"+name][HEAD][:, COORDINATES] for name in ("q", "k", "v")]
    limit = float(max(np.abs(values).max() for values in all_values))
    for ax, name, values in zip(axes, ("Queries Q", "Keys K", "Values V"), all_values):
        im = heatmap(ax, values, True, name, labels, range(1, 9), limit)
        fig.colorbar(im, ax=ax, shrink=.8, label="Signed feature")
    axes[-1].set_xlabel("Feature coordinate (first 8 of 64)")
    fig.suptitle("DistilGPT2: first block, first attention head", fontsize=13)
    path = output / "ch04_pretrained_qkv.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig = plt.figure(figsize=(8.4, 6.1), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1))
    ax_score, ax_weight = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    score = arrays["scaled_scores"]
    im = heatmap(ax_score, score, True, "(a) Scores QKᵀ / 8 before masking", labels, labels)
    fig.colorbar(im, ax=ax_score, shrink=.8)
    im = heatmap(ax_weight, arrays["original_attention"][HEAD], False,
                 "(b) Actual causal attention weights", labels, labels, 1.)
    fig.colorbar(im, ax=ax_weight, shrink=.8)
    for ax in (ax_score, ax_weight):
        ax.set(xlabel="Key / value position", ylabel="Query position")
    ax = fig.add_subplot(grid[1, :])
    vals = np.vstack((arrays["last_query_contributions"][:, :2],
                      arrays["original_head_reads"][HEAD, -1, :2]))
    x = np.arange(4)
    ax.bar(x-.18, vals[:, 0], .34, color=blue, label="Value feature 1")
    ax.bar(x+.18, vals[:, 1], .34, color=orange, label="Value feature 2")
    ax.axhline(0, color="#555555", lw=.7)
    ax.axvline(2.5, color="#999999", ls=":")
    ax.set_xticks(x, labels + ["SUM\nread at ·car"])
    ax.set(title="(c) The ·car query: weighted values, then their sum", ylabel="Signed feature contribution")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    path = output / "ch04_pretrained_read.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig = plt.figure(figsize=(8.4, 5.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1))
    for index, prefix in enumerate(("original", "changed")):
        ax = fig.add_subplot(grid[0, index])
        row_labels = report["tokens"][index]["pieces"]
        im = heatmap(ax, arrays[prefix+"_attention"][HEAD], False,
                     report["tokens"][index]["text"], row_labels, row_labels, 1.)
        ax.set(xlabel="Key position", ylabel="Query position")
        fig.colorbar(im, ax=ax, shrink=.8)
    ax = fig.add_subplot(grid[1, :])
    delta = arrays["head_read_delta"][:, COORDINATES]
    im = heatmap(ax, delta, True, "Changed read minus original read (first 8 of 64 features)",
                 ["1: the (same)", "2: ·red (same)", "3: noun (changed)"], range(1, 9), decimals=3)
    ax.set_xlabel("Feature coordinate")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle("Change only a future token: car → bike", fontsize=13)
    path = output / "ch04_pretrained_causality.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return {path.name: sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch04-pretrained"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    report, arrays = observe(args.cache_dir, args.offline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "ch04_pretrained_observations.npz"
    np.savez_compressed(archive, **arrays)
    report["source_sha256"] = sha256(__file__)
    report["attention_core_source_sha256"] = sha256(Path(__file__).with_name("attention.py"))
    report["artifacts"] = {archive.name: sha256(archive),
                           **draw_figures(report, arrays, args.output_dir)}
    (args.output_dir / "PRETRAINED_FIGURES_RUN.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
