"""Inspect RMSNorm, SwiGLU, RoPE, and residual updates in pinned SmolLM2.

All plotted quantities come from frozen pretrained weights and a fixed input.
Independent float64 calculations check the native float32 module outputs.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import numpy as np


MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
TEXT = "The red car stopped near the bridge."
POSITION = 2
KEY_POSITION = 1
HEAD = 0
PAIR = 0
FEATURES = tuple(range(8))  # Fixed before inspecting any values.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def linear64(row, weight):
    return np.asarray(row, dtype=np.float64) @ weight.astype(np.float64).T


def rms64(x, gain, epsilon):
    x = x.astype(np.float64)
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+epsilon) * gain


def sigmoid64(x):
    x = np.asarray(x, dtype=np.float64)
    output = np.empty_like(x)
    positive = x >= 0
    output[positive] = 1/(1+np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    output[~positive] = exp_x/(1+exp_x)
    return output


def pieces(tokenizer, ids):
    return [tokenizer.decode([int(i)]).replace(" ", "·").replace("\n", "\\n") for i in ids]


def observe(cache_dir, offline):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, LlamaForCausalLM
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    common = dict(revision=REVISION, cache_dir=str(cache_dir), token=False,
                  trust_remote_code=False, local_files_only=offline)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, **common)
    model, loading = LlamaForCausalLM.from_pretrained(
        MODEL, **common, use_safetensors=True, attn_implementation="eager", output_loading_info=True)
    assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    assert model.config._commit_hash == REVISION
    model.eval().float().requires_grad_(False)
    config = model.config
    assert (config.hidden_size, config.intermediate_size, config.num_hidden_layers) == (576, 1536, 30)
    assert config.hidden_act == "silu" and config.rope_scaling is None
    assert not config.attention_bias and not config.mlp_bias
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    encoded = tokenizer(TEXT, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"][0].numpy()
    assert tokenizer.decode(encoded["input_ids"][0]) == TEXT
    assert tokenizer.decode([int(ids[POSITION])]) == " car"
    assert tokenizer.decode([int(ids[KEY_POSITION])]) == " red"
    length = len(ids)
    positions = torch.arange(length)[None, :]
    first = model.model.layers[0]
    captured, raw_blocks = {}, []
    handles = []

    def hook(name):
        def record(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name+"_input"] = args[0].detach().cpu().numpy().copy()
            captured[name] = value.detach().cpu().numpy().copy()
        return record

    for name, module in (("attention_norm", first.input_layernorm),
                         ("ffn_norm", first.post_attention_layernorm),
                         ("gate", first.mlp.gate_proj), ("up", first.mlp.up_proj),
                         ("gate_activated", first.mlp.act_fn),
                         ("down", first.mlp.down_proj), ("mlp", first.mlp),
                         ("q", first.self_attn.q_proj), ("k", first.self_attn.k_proj),
                         ("v", first.self_attn.v_proj)):
        handles.append(module.register_forward_hook(hook(name)))

    def attention_hook(module, args, kwargs, output):
        captured["attention"] = output[0].detach().numpy().copy()
        captured["cos"] = kwargs["position_embeddings"][0].detach().numpy().copy()
        captured["sin"] = kwargs["position_embeddings"][1].detach().numpy().copy()

    handles.append(first.self_attn.register_forward_hook(attention_hook, with_kwargs=True))
    for block in model.model.layers:
        handles.append(block.register_forward_hook(
            lambda module, args, output: raw_blocks.append(output.detach().numpy().copy())))
    with torch.inference_mode():
        native = model(**encoded, position_ids=positions, output_attentions=True,
                       output_hidden_states=True, use_cache=False)
    for handle in handles:
        handle.remove()

    arrays = {name: value[0] for name, value in captured.items()}
    arrays["input_ids"] = ids
    arrays["block_outputs"] = np.stack(raw_blocks)[:, 0]
    arrays["all_attention"] = np.stack([a[0].numpy() for a in native.attentions])
    arrays["final_hidden"] = native.hidden_states[-1][0].numpy()
    arrays["logits"] = native.logits[0].numpy()
    assert len(raw_blocks) == config.num_hidden_layers
    np.testing.assert_allclose(arrays["all_attention"].sum(-1), 1, atol=4e-7, rtol=0)
    assert np.count_nonzero(np.triu(arrays["all_attention"], 1)) == 0

    epsilon = first.input_layernorm.variance_epsilon
    gain = first.input_layernorm.weight.numpy()
    h = arrays["attention_norm_input"][POSITION]
    normalized = h.astype(np.float64)/np.sqrt(np.mean(h.astype(np.float64)**2)+epsilon)
    reconstructed_norm = rms64(arrays["attention_norm_input"], gain, epsilon)
    np.testing.assert_allclose(reconstructed_norm, arrays["attention_norm"], atol=5e-6, rtol=2e-5)
    arrays["norm_gain"] = gain
    arrays["norm_before_gain"] = normalized
    reconstructed_ffn_norm = rms64(arrays["ffn_norm_input"], first.post_attention_layernorm.weight.numpy(), epsilon)
    np.testing.assert_allclose(reconstructed_ffn_norm, arrays["ffn_norm"], atol=5e-6, rtol=2e-5)

    u = arrays["ffn_norm"][POSITION]
    gate = linear64(u, first.mlp.gate_proj.weight.numpy())
    up = linear64(u, first.mlp.up_proj.weight.numpy())
    activated = gate*sigmoid64(gate)
    product = activated*up
    ffn = linear64(product, first.mlp.down_proj.weight.numpy())
    for independent, actual in ((gate, arrays["gate"][POSITION]), (up, arrays["up"][POSITION]),
                                (product, arrays["down_input"][POSITION]), (ffn, arrays["mlp"][POSITION])):
        np.testing.assert_allclose(independent, actual, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(activated, arrays["gate_activated"][POSITION], atol=3e-5, rtol=3e-5)
    arrays["down_contributions_feature0"] = arrays["down_input"][POSITION].astype(np.float64)*first.mlp.down_proj.weight[0].numpy()
    np.testing.assert_allclose(arrays["down_contributions_feature0"].sum(), arrays["mlp"][POSITION, 0], atol=3e-5, rtol=0)

    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    width = config.hidden_size//heads
    groups = heads//kv_heads
    q = torch.from_numpy(captured["q"]).reshape(1, length, heads, width).transpose(1, 2)
    k = torch.from_numpy(captured["k"]).reshape(1, length, kv_heads, width).transpose(1, 2)
    with torch.inference_mode():
        qr, kr = apply_rotary_pos_emb(q, k, torch.from_numpy(captured["cos"]), torch.from_numpy(captured["sin"]))
    q, k, qr, kr = [value[0].numpy() for value in (q, k, qr, kr)]
    frequencies = config.rope_theta**(-2*np.arange(width//2, dtype=np.float64)/width)

    def independent_rotate(values, indices):
        angles = np.asarray(indices, dtype=np.float64)[:, None]*frequencies
        a, b = np.split(values.astype(np.float64), 2, axis=-1)
        return np.concatenate((a*np.cos(angles)-b*np.sin(angles),
                               a*np.sin(angles)+b*np.cos(angles)), axis=-1)

    np.testing.assert_allclose(independent_rotate(q, np.arange(length)), qr, atol=4e-6, rtol=2e-5)
    np.testing.assert_allclose(independent_rotate(k, np.arange(length)), kr, atol=4e-6, rtol=2e-5)
    np.testing.assert_allclose(np.linalg.norm(qr, axis=-1), np.linalg.norm(q, axis=-1), atol=5e-6, rtol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(kr, axis=-1), np.linalg.norm(k, axis=-1), atol=5e-6, rtol=2e-6)
    scores = qr.astype(np.float64) @ np.repeat(kr, groups, axis=0).astype(np.float64).transpose(0, 2, 1)/math.sqrt(width)
    masked = np.where(np.tril(np.ones((length, length), dtype=bool)), scores, -np.inf)
    probability = np.exp(masked-masked.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    np.testing.assert_allclose(probability, arrays["all_attention"][0], atol=3e-6, rtol=2e-5)
    values = captured["v"][0].reshape(length, kv_heads, width).transpose(1, 0, 2)
    attended = (probability @ np.repeat(values, groups, axis=0).astype(np.float64)).transpose(1, 0, 2).reshape(length, -1)
    projected = linear64(attended, first.self_attn.o_proj.weight.numpy())
    np.testing.assert_allclose(projected, arrays["attention"], atol=3e-5, rtol=3e-5)
    key_head = HEAD//groups
    rotated_query, rotated_key = qr[HEAD, POSITION], kr[key_head, KEY_POSITION]
    pairs = rotated_query[:width//2]*rotated_key[:width//2]+rotated_query[width//2:]*rotated_key[width//2:]
    np.testing.assert_allclose(pairs.sum()/math.sqrt(width), scores[HEAD, POSITION, KEY_POSITION], atol=3e-6, rtol=0)
    shifted_scores = independent_rotate(q, np.arange(length)+7) @ np.repeat(
        independent_rotate(k, np.arange(length)+7), groups, axis=0).transpose(0, 2, 1)/math.sqrt(width)
    original_scores64 = independent_rotate(q, np.arange(length)) @ np.repeat(
        independent_rotate(k, np.arange(length)), groups, axis=0).transpose(0, 2, 1)/math.sqrt(width)
    np.testing.assert_allclose(shifted_scores, original_scores64, atol=1e-11, rtol=0)
    arrays.update(q_unrotated=q, k_unrotated=k, q_rotated=qr, k_rotated=kr,
                  head_scores=scores, pair_contributions=pairs, frequencies=frequencies)

    after_attention = arrays["attention_norm_input"]+arrays["attention"]
    np.testing.assert_array_equal(after_attention, arrays["ffn_norm_input"])
    np.testing.assert_array_equal(after_attention+arrays["mlp"], arrays["block_outputs"][0])
    stream = np.concatenate((arrays["attention_norm_input"][None], arrays["block_outputs"]), axis=0)
    arrays["residual_rms"] = np.sqrt(np.mean(stream.astype(np.float64)**2, axis=-1))
    final_norm = rms64(arrays["block_outputs"][-1], model.model.norm.weight.numpy(), model.model.norm.variance_epsilon)
    np.testing.assert_allclose(final_norm, arrays["final_hidden"], atol=5e-5, rtol=3e-5)

    with torch.inference_mode():
        shifted = model(**encoded, position_ids=positions+7, use_cache=False)
        changed_ids = encoded["input_ids"].clone()
        changed_ids[0, -1] = tokenizer.encode("!", add_special_tokens=False)[0]
        assert changed_ids[0, -1] != encoded["input_ids"][0, -1]
        changed = model(input_ids=changed_ids, attention_mask=encoded["attention_mask"],
                        position_ids=positions, use_cache=False)
    np.testing.assert_allclose(shifted.logits.numpy(), native.logits.numpy(), atol=1e-4, rtol=0)
    np.testing.assert_array_equal(changed.logits[0, :-1].numpy(), arrays["logits"][:-1])
    weights = hf_hub_download(MODEL, "model.safetensors", revision=REVISION,
                              cache_dir=str(cache_dir), local_files_only=True, token=False)
    report = {"model": MODEL, "revision": REVISION, "weights_sha256": sha256(weights),
              "parameters": sum(p.numel() for p in model.parameters()), "loading": loading,
              "configuration": {name: getattr(config, name) for name in
                  ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                   "num_key_value_heads", "rms_norm_eps", "rope_theta", "max_position_embeddings", "vocab_size")},
              "text": TEXT, "input_ids": ids.tolist(), "pieces": pieces(tokenizer, ids),
              "probe": {"array_position": POSITION, "key_position": KEY_POSITION, "query_head": HEAD,
                        "key_head": key_head, "rotary_pair": PAIR, "pair_features": [PAIR, PAIR+width//2],
                        "displayed_features": list(FEATURES)},
              "normalization": {"mean_square": float(np.mean(h.astype(np.float64)**2)),
                  "denominator": float(np.sqrt(np.mean(h.astype(np.float64)**2)+epsilon)),
                  "input_first_feature": float(h[0]), "gain_first_feature": float(gain[0]),
                  "output_first_feature": float(arrays["attention_norm"][POSITION, 0]),
                  "rms_before_gain": float(np.sqrt(np.mean(normalized**2))),
                  "rms_after_gain": float(np.sqrt(np.mean(arrays["attention_norm"][POSITION].astype(np.float64)**2))),
                  "reconstruction_max_error": float(np.max(np.abs(reconstructed_norm-arrays["attention_norm"])))},
              "swiglu": {"first_gate": float(arrays["gate"][POSITION, 0]),
                  "first_up": float(arrays["up"][POSITION, 0]),
                  "first_activated_gate": float(arrays["gate_activated"][POSITION, 0]),
                  "first_product": float(arrays["down_input"][POSITION, 0]),
                  "activated_negative_count": int((arrays["gate_activated"][POSITION] < 0).sum()),
                  "activated_above_one_count": int((arrays["gate_activated"][POSITION] > 1).sum()),
                  "first_eight_down_sum": float(arrays["down_contributions_feature0"][:8].sum()),
                  "full_down_sum": float(arrays["down_contributions_feature0"].sum()),
                  "native_output_first_feature": float(arrays["mlp"][POSITION, 0]),
                  "output_reconstruction_max_error": float(np.max(np.abs(ffn-arrays["mlp"][POSITION])))},
              "rope": {"head_width": width, "relative_angle_pair0": float((KEY_POSITION-POSITION)*frequencies[0]),
                  "pair0_dot": float(pairs[0]), "all_pairs_dot": float(pairs.sum()),
                  "scaled_score": float(scores[HEAD, POSITION, KEY_POSITION]),
                  "native_attention_row": arrays["all_attention"][0, HEAD, POSITION].tolist(),
                  "attention_reconstruction_max_error": float(np.max(np.abs(probability-arrays["all_attention"][0]))),
                  "attention_output_reconstruction_max_error": float(np.max(np.abs(projected-arrays["attention"]))),
                  "shared_shift_scores64_max_error": float(np.max(np.abs(shifted_scores-original_scores64))),
                  "shared_shift_model_logits_max_error": float(np.max(np.abs(shifted.logits.numpy()-native.logits.numpy())))},
              "residual": {"additions_match_native_exactly": True,
                  "stage_rms": arrays["residual_rms"][:, POSITION].tolist(),
                  "final_norm_rms": float(np.sqrt(np.mean(arrays["final_hidden"][POSITION].astype(np.float64)**2))),
                  "future_change_earlier_logits_max_error": float(np.max(np.abs(changed.logits[0, :-1].numpy()-arrays["logits"][:-1])))}}
    del model
    return report, arrays


def figures(report, arrays, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    files = []

    def save(fig, name):
        path = directory/name
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        files.append(path)

    def matrix(ax, data, rows, title):
        scale = float(np.max(np.abs(data))) or 1
        ax.imshow(data, cmap="PuOr_r", vmin=-scale, vmax=scale, aspect="auto")
        ax.set_xticks(range(8), FEATURES)
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel("Feature array index (first 8 shown)")
        ax.set_title(title, loc="left", fontsize=11, pad=12)
        for i in range(len(rows)):
            for j in range(8):
                value = data[i, j]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > .67*scale else "#222222")

    p = POSITION
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.4), gridspec_kw={"height_ratios": [1.5, 1]}, layout="constrained")
    rows = np.stack((arrays["attention_norm_input"][p, :8], arrays["norm_before_gain"][:8],
                     arrays["norm_gain"][:8], arrays["attention_norm"][p, :8]))
    matrix(axes[0], rows, ["Input", "Before gains", "Learned gains", "After gains"],
           "(a) SmolLM2 block 1: attention-input RMSNorm at '·car'")
    norm = report["normalization"]
    magnitudes = [math.sqrt(norm["mean_square"]), norm["rms_before_gain"], norm["rms_after_gain"]]
    bars = axes[1].barh(["Raw input", "Before gains", "After gains"], magnitudes, color=["#64748b", "#2985ad", "#8e5ea2"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(magnitudes)*1.25)
    axes[1].bar_label(bars, fmt="%.5f", padding=4)
    axes[1].set_xlabel("Root mean square across all 576 features")
    axes[1].set_title("(b) Learned gains change the output scale", loc="left", fontsize=11)
    save(fig, "ch08_pretrained_rmsnorm.png")

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.7), layout="constrained")
    rows = np.stack((arrays["gate"][p, :8], arrays["up"][p, :8],
                     arrays["gate_activated"][p, :8], arrays["down_input"][p, :8]))
    matrix(axes[0], rows, ["Gate projection g", "Up projection u", "SiLU(g)", "SiLU(g) × u"],
           "(a) SmolLM2 block 1: first 8 of 1,536 FFN features")
    cumulative = np.cumsum(arrays["down_contributions_feature0"])
    axes[1].plot(np.arange(1, len(cumulative)+1), cumulative, color="#257ca3", linewidth=1.4)
    axes[1].scatter([len(cumulative)], [cumulative[-1]], color="#178473", zorder=3)
    axes[1].axhline(report["swiglu"]["native_output_first_feature"], color="#178473", linestyle="--", label="Native FFN output, feature 0")
    axes[1].set_xlabel("Number of intermediate contributions summed")
    axes[1].set_ylabel("Cumulative contribution\nto output feature 0")
    axes[1].set_title("(b) The down projection sums all 1,536 contributions", loc="left", fontsize=11)
    axes[1].legend(loc="best", fontsize=8)
    save(fig, "ch08_pretrained_swiglu.png")

    fig = plt.figure(figsize=(8.4, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    ax = fig.add_subplot(grid[0, 0])
    vectors = [(arrays["q_unrotated"][HEAD, POSITION], "Q before", "#2782ad", ":"),
               (arrays["q_rotated"][HEAD, POSITION], "Q rotated", "#2782ad", "-"),
               (arrays["k_unrotated"][report["probe"]["key_head"], KEY_POSITION], "K before", "#d98925", ":"),
               (arrays["k_rotated"][report["probe"]["key_head"], KEY_POSITION], "K rotated", "#d98925", "-")]
    a, b = report["probe"]["pair_features"]
    maximum = max(abs(float(v[i])) for v, *_ in vectors for i in (a, b))*1.25
    for vector, label, color, style in vectors:
        ax.annotate("", xy=(vector[a], vector[b]), xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": color, "linestyle": style, "lw": 1.7})
        ax.plot([], [], color=color, linestyle=style, label=label)
    ax.axhline(0, color="#cccccc", linewidth=.6)
    ax.axvline(0, color="#cccccc", linewidth=.6)
    ax.set(xlim=(-maximum, maximum), ylim=(-maximum, maximum), aspect="equal",
           xlabel=f"Feature {a}", ylabel=f"Feature {b}")
    ax.set_title("(a) Native pair 0: Q at 2, K at 1", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax = fig.add_subplot(grid[0, 1])
    contributions = arrays["pair_contributions"]
    ax.bar(np.arange(len(contributions)), contributions,
           color=np.where(contributions >= 0, "#2985ad", "#be773b"), width=.82)
    ax.axhline(0, color="#444444", linewidth=.6)
    ax.set_xlabel("Rotary pair index (all 32 pairs)")
    ax.set_ylabel("Pair contribution to Q · K")
    ax.set_title("(b) Sum pairs, then divide by √64", loc="left", fontsize=10)
    ax.text(.98, .98, f"Sum = {contributions.sum():.5f}\nScaled score = {report['rope']['scaled_score']:.5f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    ax = fig.add_subplot(grid[1, :])
    row = np.asarray(report["rope"]["native_attention_row"])
    bars = ax.bar(np.arange(len(row)), row, color=["#2985ad" if i <= POSITION else "#cbd5e1" for i in range(len(row))])
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xticks(np.arange(len(row)), report["pieces"])
    ax.set_ylim(0, min(1.12, max(row)*1.28))
    ax.set_ylabel("Attention probability")
    ax.set_xlabel("Input key position; later keys are masked")
    ax.set_title("(c) Native head 1, query '·car': softmax across allowed keys", loc="left", fontsize=11)
    save(fig, "ch08_pretrained_rope.png")

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.7), layout="constrained")
    h, a, b = arrays["attention_norm_input"][p], arrays["attention"][p], arrays["mlp"][p]
    rows = np.stack((h[:8], a[:8], (h+a)[:8], b[:8], (h+a+b)[:8]))
    matrix(axes[0], rows, ["Residual H", r"Attention $\Delta_a$", r"H + $\Delta_a$",
                          r"FFN $\Delta_f$", r"H + $\Delta_a$ + $\Delta_f$"],
           "(a) Two actual additions in SmolLM2 block 1 at '·car'")
    rms = arrays["residual_rms"][:, p]
    axes[1].plot(np.arange(len(rms)), rms, color="#2985ad", marker="o", markersize=3, label="Residual stream")
    axes[1].plot([len(rms)-1, len(rms)], [rms[-1], report["residual"]["final_norm_rms"]], color="#8e5ea2", linestyle="--")
    axes[1].scatter([len(rms)], [report["residual"]["final_norm_rms"]], marker="*", s=90, color="#8e5ea2", label="Separate final RMSNorm")
    axes[1].set_xticks([0, 5, 10, 15, 20, 25, 30, 31], ["0", "5", "10", "15", "20", "25", "30", "Final\nnorm"])
    axes[1].get_xticklabels()[-2].set_ha("right")
    axes[1].get_xticklabels()[-1].set_ha("left")
    axes[1].set_xlabel("Completed blocks; 0 is the token embedding")
    axes[1].set_ylabel("RMS over 576 features")
    axes[1].set_title("(b) The residual stream is not forced to unit RMS", loc="left", fontsize=11)
    axes[1].legend(fontsize=8)
    save(fig, "ch08_pretrained_residual.png")
    return {path.name: sha256(path) for path in files}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch08-pretrained"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report, arrays = observe(args.cache_dir, args.offline)
    report["execution"] = "Frozen CPU float32 inference; eager attention; one thread; no training or sampling"
    report["selection"] = "Fixed prompt; block 1; query head 1; positions 2 and 1; pair 0; first eight coordinates"
    report["software"] = {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in
        ("torch", "transformers", "tokenizers", "huggingface_hub", "safetensors", "numpy", "matplotlib", "Pillow")}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir/"observations.npz"
    np.savez_compressed(archive, **arrays)
    report["artifacts"] = {archive.name: sha256(archive), **figures(report, arrays, args.output_dir)}
    report["source_sha256"] = sha256(__file__)
    (args.output_dir/"PRETRAINED_FIGURES_RUN.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
