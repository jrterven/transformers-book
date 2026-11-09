"""Inspect heads, post-norm residuals, and a ReLU FFN in the trained model.

Uses Chapter 6's saved color-reversal checkpoint and Chapter 5's actual modules.
No new training, token sampling, or external weights are needed for inspection.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch

from examples.ch04_attention.attention import scaled_dot_product_attention
from examples.ch06_training.data import EOS, VOCAB, collate, make_splits
from examples.ch06_training.decoding import greedy_decode
from examples.ch06_training.training import load_checkpoint


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def observations(checkpoint):
    torch.set_num_threads(1)
    model, saved = load_checkpoint(checkpoint)
    model.eval().requires_grad_(False)
    expected = {"d_model": 64, "n_heads": 4, "d_ff": 128,
                "n_encoder_layers": 2, "n_decoder_layers": 2, "dropout": 0.0}
    if any(saved["model_config"][key] != value for key, value in expected.items()):
        raise ValueError("these figure layouts require Chapter 6's reference architecture")
    if saved["step"] <= 0:
        raise ValueError("inspection requires a checkpoint after training updates")
    experiment = saved["experiment"]
    sizes = experiment["split_sizes"]
    splits = make_splits(seed=experiment["data_seed"], n_train=sizes[0],
                         n_validation=sizes[1], n_test=sizes[2],
                         min_length=experiment["min_length"], max_length=experiment["max_length"])
    for name, split in zip(("train", "validation", "test"), splits):
        fingerprint = hashlib.sha256(json.dumps(split, separators=(",", ":")).encode()).hexdigest()
        if fingerprint != experiment["split_sha256"][name]:
            raise ValueError(f"reconstructed {name} split differs from the checkpoint")
    _, _, test = splits
    index = next(i for i, sequence in enumerate(test) if len(sequence) == 4)
    sequence = test[index]
    batch = collate([sequence])
    # Generate before hooks are installed, then inspect supplied reference inputs.
    generated = greedy_decode(model, batch.source, max_new_tokens=8, cached=False)[0].tolist()
    generated = generated[:generated.index(EOS)+1] if EOS in generated else generated
    captured = {}

    def hook(name):
        def capture(module, inputs, output):
            captured[name] = (tuple(x.detach().clone() if torch.is_tensor(x) else x
                                    for x in inputs),
                              tuple(x.detach().clone() for x in output)
                              if isinstance(output, tuple) else output.detach().clone())
        return capture

    encoder, decoder = model.encoder_layers[0], model.decoder_layers[0]
    handles = [module.register_forward_hook(hook(name)) for name, module in (
        ("encoder_attention", encoder.self_attention), ("norm1", encoder.norm1),
        ("ffn", encoder.feed_forward), ("expand", encoder.feed_forward.expand),
        ("contract", encoder.feed_forward.contract), ("cross", decoder.cross_attention))]
    try:
        logits, _ = model(batch.source, batch.decoder_input, return_weights=True)
    finally:
        for handle in handles:
            handle.remove()

    # Cross-attention: reconstruct four head reads, concatenate, project, and
    # independently decompose the projected result into per-head contributions.
    cross = decoder.cross_attention
    query, memory, _, allowed = captured["cross"][0]
    native_cross, native_weights = captured["cross"][1]
    q = cross.split_heads(cross.q_proj(query))[0].numpy().astype(np.float64)
    k = cross.split_heads(cross.k_proj(memory))[0].numpy().astype(np.float64)
    v = cross.split_heads(cross.v_proj(memory))[0].numpy().astype(np.float64)
    reads, weights64 = scaled_dot_product_attention(q, k, v, allowed[0].numpy())
    np.testing.assert_allclose(weights64, native_weights[0].numpy(), atol=2e-6, rtol=2e-6)
    combined = reads.transpose(1, 0, 2).reshape(reads.shape[1], -1)
    output_weight = cross.out_proj.weight.numpy().T.astype(np.float64)
    reconstructed = combined @ output_weight
    np.testing.assert_allclose(reconstructed, native_cross[0].numpy(), atol=2e-5, rtol=3e-6)
    head_width = cross.head_width
    projected_heads = np.stack([
        reads[h] @ output_weight[h*head_width:(h+1)*head_width]
        for h in range(cross.n_heads)])
    np.testing.assert_allclose(projected_heads.sum(0), reconstructed, atol=1e-12)
    np.testing.assert_allclose(native_weights[0].sum(-1).numpy(), 1, atol=2e-7, rtol=0)

    # Encoder token position 2 (one-based), selected before viewing any feature.
    position = 1
    incoming = captured["encoder_attention"][0][0][0].numpy().astype(np.float64)
    update = captured["encoder_attention"][1][0][0].numpy().astype(np.float64)
    residual = captured["norm1"][0][0][0].numpy().astype(np.float64)
    np.testing.assert_allclose(incoming + update, residual, atol=2e-6, rtol=2e-6)
    mean = residual.mean(-1, keepdims=True)
    variance = residual.var(-1, keepdims=True)
    standardized = (residual - mean) / np.sqrt(variance + encoder.norm1.eps)
    gamma = encoder.norm1.weight.numpy().astype(np.float64)
    beta = encoder.norm1.bias.numpy().astype(np.float64)
    normalized = standardized * gamma + beta
    native_normalized = captured["norm1"][1][0].numpy().astype(np.float64)
    np.testing.assert_allclose(normalized, native_normalized, atol=3e-6, rtol=3e-6)

    # FFN: reconstruct both affine maps and their intervening ReLU independently.
    ffn = encoder.feed_forward
    ffn_input = captured["ffn"][0][0][0].numpy().astype(np.float64)
    expanded = ffn_input @ ffn.expand.weight.numpy().T + ffn.expand.bias.numpy()
    activated = np.maximum(expanded, 0)
    contracted = activated @ ffn.contract.weight.numpy().T + ffn.contract.bias.numpy()
    np.testing.assert_allclose(expanded, captured["expand"][1][0].numpy(), atol=4e-6, rtol=3e-6)
    np.testing.assert_allclose(activated, captured["contract"][0][0][0].numpy(), atol=4e-6, rtol=3e-6)
    native_ffn = captured["ffn"][1][0].numpy().astype(np.float64)
    np.testing.assert_allclose(contracted, native_ffn, atol=1e-5, rtol=3e-6)
    changed_input = captured["ffn"][0][0].clone()
    changed_input[:, 0] += 1  # Local FFN intervention, not a new encoder pass.
    changed_output = ffn(changed_input)
    torch.testing.assert_close(changed_output[:, 1:], captured["ffn"][1][:, 1:], atol=0, rtol=0)

    arrays = {
        "source_ids": batch.source[0].numpy(),
        "decoder_input_ids": batch.decoder_input[0].numpy(),
        "target_ids": batch.target[0].numpy(), "logits": logits[0].numpy(),
        "cross_q": q, "cross_k": k, "cross_v": v,
        "cross_weights": native_weights[0].numpy(), "cross_head_reads": reads,
        "cross_output_weight": output_weight, "cross_projected_heads": projected_heads,
        "cross_native_output": native_cross[0].numpy(),
        "encoder_input": incoming, "encoder_attention_update": update,
        "encoder_residual": residual, "encoder_standardized": standardized,
        "encoder_norm_gamma": gamma, "encoder_norm_beta": beta,
        "encoder_normalized": native_normalized,
        "ffn_input": ffn_input, "ffn_expanded": expanded, "ffn_activated": activated,
        "ffn_output": native_ffn,
    }
    parameter_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        parameter_hash.update(name.encode())
        parameter_hash.update(value.numpy().tobytes())
    row = position
    report = {
        "checkpoint_sha256": sha256(checkpoint), "parameter_sha256": parameter_hash.hexdigest(),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "selected_update": saved["step"], "configuration": saved["model_config"],
        "experiment": saved["experiment"], "selected_validation": saved["validation"],
        "mode": "frozen CPU model in eval mode; reference-conditioned observation; no new training",
        "vocabulary": list(VOCAB), "test_index": index,
        "source": [VOCAB[t] for t in sequence],
        "decoder_inputs": [VOCAB[t] for t in batch.decoder_input[0].tolist()],
        "target": [VOCAB[t] for t in batch.target[0].tolist()],
        "generated": [VOCAB[t] for t in generated],
        "selection": "first length-four test source, before inspecting predictions or maps",
        "display": {"layer": 1, "attention_heads": [1, 2], "all_projected_heads": cross.n_heads,
                    "cross_query_position": 2, "encoder_token_position": row+1,
                    "model_features": list(range(1, 9)), "expanded_features": list(range(1, 17))},
        "normalization": {"mean": float(mean[row, 0]), "variance": float(variance[row, 0]),
                          "epsilon": encoder.norm1.eps, "residual_first": float(residual[row, 0]),
                          "standardized_first": float(standardized[row, 0]),
                          "gamma_first": float(gamma[0]), "beta_first": float(beta[0]),
                          "output_first": float(native_normalized[row, 0]),
                          "output_mean": float(native_normalized[row].mean()),
                          "output_variance": float(native_normalized[row].var())},
        "ffn": {"active_expanded_features": int(np.count_nonzero(activated[row])),
                "expanded_width": activated.shape[-1], "output_width": native_ffn.shape[-1],
                "output_first": float(native_ffn[row, 0]), "unchanged_other_rows_max_difference": 0.},
        "checks": {"max_attention_error": float(np.abs(weights64-native_weights[0].numpy()).max()),
                   "max_projected_read_error": float(np.abs(reconstructed-native_cross[0].numpy()).max()),
                   "max_norm_error": float(np.abs(normalized-native_normalized).max()),
                   "max_ffn_error": float(np.abs(contracted-native_ffn).max())},
        "software": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                     for name in ("torch", "numpy", "matplotlib", "Pillow")}},
    }
    return report, arrays


def figures(report, arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    paths = []
    row = report["display"]["encoder_token_position"] - 1

    def heatmap(ax, data, title, rows, columns, signed=False, limit=None, decimals=2):
        limit = float(np.abs(data).max()) if limit is None else limit
        im = ax.imshow(data, aspect="auto", cmap="PuOr_r" if signed else "Blues",
                       vmin=-limit if signed else 0, vmax=limit)
        ax.set_xticks(range(data.shape[1]), columns)
        ax.set_yticks(range(data.shape[0]), rows)
        ax.set_title(title, loc="left", pad=10)
        for (r, c), value in np.ndenumerate(data):
            displayed = 0.0 if round(float(value), decimals) == 0 else value
            ax.text(c, r, f"{displayed:.{decimals}f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > .65 * limit else "#222222")
        return im

    fig = plt.figure(figsize=(8.4, 6.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1))
    source_labels = [f"{i+1}: {word}" for i, word in enumerate(report["source"])]
    target_labels = [f"{i+1}: {word.strip('<>')}" for i, word in enumerate(report["decoder_inputs"])]
    for head in (0, 1):
        ax = fig.add_subplot(grid[0, head])
        im = heatmap(ax, arrays["cross_weights"][head], f"Cross-attention head {head+1}",
                     target_labels, source_labels, limit=1)
        ax.tick_params(axis="x", rotation=25)
        ax.set(xlabel="Source key position", ylabel="Decoder input position")
        fig.colorbar(im, ax=ax, shrink=.8)
    ax = fig.add_subplot(grid[1, :])
    query = report["display"]["cross_query_position"] - 1
    components = arrays["cross_projected_heads"][:, query, :8]
    data = np.vstack((components, arrays["cross_native_output"][query, :8]))
    im = heatmap(ax, data, "Output projection: four signed contributions and their sum",
                 ["Head 1", "Head 2", "Head 3", "Head 4", "SUM"], range(1, 9), signed=True)
    ax.set_xlabel("Model feature (first 8 of 64), decoder query 2")
    fig.colorbar(im, ax=ax, shrink=.8)
    path = output / "ch05_trained_heads.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.9), layout="constrained")
    x = np.arange(8)
    axes[0].bar(x-.18, arrays["encoder_input"][row, :8], .34, color="#2468a0", label="Incoming vector")
    axes[0].bar(x+.18, arrays["encoder_attention_update"][row, :8], .34, color="#c8772f", label="Attention update")
    axes[0].plot(x, arrays["encoder_residual"][row, :8], "D", color="#222222", ms=5, label="Their sum")
    axes[0].axhline(0, color="#555555", lw=.7)
    axes[0].set_xticks(x, range(1, 9))
    axes[0].set(title="(a) Add the unchanged input to its learned update", ylabel="Signed feature value")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    data = np.stack((arrays["encoder_standardized"][row, :8], arrays["encoder_normalized"][row, :8]))
    im = heatmap(axes[1], data, "(b) Normalize the sum, then apply learned scale and offset",
                 ["Standardized", "After gamma / beta"], range(1, 9), signed=True, decimals=3)
    axes[1].set_xlabel("Model feature (first 8 of 64); statistics use all 64 features")
    fig.colorbar(im, ax=axes[1], shrink=.8)
    fig.suptitle(f"First encoder layer, source token 2: {report['source'][1]}", fontsize=13)
    path = output / "ch05_trained_postnorm.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.5), layout="constrained")
    data = np.stack((arrays["ffn_expanded"][row, :16], arrays["ffn_activated"][row, :16]))
    im = heatmap(axes[0], data, "(a) ReLU keeps positive activations and zeros negative ones",
                 ["Before ReLU", "After ReLU"], range(1, 17), signed=True, decimals=1)
    axes[0].set_xlabel("Expanded feature (first 16 of 128)")
    fig.colorbar(im, ax=axes[0], shrink=.8)
    values = arrays["ffn_output"][row, :8]
    axes[1].bar(range(8), values, color="#168477")
    axes[1].set_xticks(range(8), range(1, 9))
    axes[1].axhline(0, color="#555555", lw=.7)
    axes[1].set(title="(b) The second learned affine map returns to width 64",
                xlabel="Output feature (first 8 of 64)", ylabel="Signed FFN update")
    fig.suptitle(f"A real positionwise FFN: 64 → 128 → 64 features", fontsize=13)
    path = output / "ch05_trained_ffn.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return {p.name: sha256(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch05-inspection"))
    args = parser.parse_args()
    report, arrays = observations(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "observations.npz"
    np.savez_compressed(archive, **arrays)
    report["artifacts"] = {archive.name: sha256(archive), **figures(report, arrays, args.output_dir)}
    report["source_sha256"] = {str(p.relative_to(Path.cwd())): sha256(p) for p in
        (Path(__file__).resolve(), Path(__file__).with_name("model.py").resolve())}
    (args.output_dir / "TRAINED_FIGURES_RUN.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
