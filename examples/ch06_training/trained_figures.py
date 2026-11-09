"""Observe a real training update, learning snapshots, and a growing KV cache.

Replay the saved Chapter 6 recipe with the canonical train_step. Verify that
the selected replay state exactly matches the supplied checkpoint before
plotting its generation and caches. No external weights or data are needed.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch

from examples.ch05_transformer.model import Transformer
from .data import BOS, EOS, PAD, VOCAB, collate, make_splits, words
from .decoding import CachedDecoder, decode_prefix, encode_source, greedy_decode
from .training import evaluate, load_checkpoint, train_step


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def snapshot(model, probe, step, cap):
    model.eval()
    logits = model(probe.source, probe.decoder_input)
    log_probabilities = logits.log_softmax(-1)
    selected = log_probabilities.gather(-1, probe.target[..., None])[0, :, 0]
    generated = greedy_decode(model, probe.source, max_new_tokens=cap, cached=False)
    predictions = logits.argmax(-1)[0]
    record = {"step": step, "state_sha256": state_sha256(model),
              "reference_conditioned_nll": float(-selected.mean()),
              "correct_next_tokens": int((predictions == probe.target[0]).sum()),
              "target_tokens": probe.target.numel(),
              "teacher_forced_choices": " ".join(VOCAB[t] for t in predictions.tolist()),
              "teacher_forced_choice_ids": predictions.tolist(),
              "generated": words(generated[0], include_special=True),
              "generated_ids": generated[0].tolist()}
    return record, selected.exp().numpy(), logits[0].numpy()


def replay(checkpoint):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reference, saved = load_checkpoint(checkpoint)
    config, experiment = saved["model_config"], saved["experiment"]
    if experiment["steps"] != 2000 or saved["step"] != 2000:
        raise ValueError("the published layouts use the 2,000-update reference run")
    if config["d_model"] != 64 or config["n_decoder_layers"] != 2 or config["dropout"] != 0:
        raise ValueError("the published layouts require Chapter 6's reference architecture")
    sizes = experiment["split_sizes"]
    splits = make_splits(seed=experiment["data_seed"], n_train=sizes[0],
                         n_validation=sizes[1], n_test=sizes[2],
                         min_length=experiment["min_length"], max_length=experiment["max_length"])
    for name, split in zip(("train", "validation", "test"), splits):
        digest = hashlib.sha256(json.dumps(split, separators=(",", ":")).encode()).hexdigest()
        if digest != experiment["split_sha256"][name]:
            raise ValueError(f"reconstructed {name} split differs from saved metadata")
    train, validation, _ = splits
    # Choose the probe before training or inspecting predictions; never use test
    # outcomes to choose the example or a replay checkpoint.
    probe_index = next(i for i, sequence in enumerate(validation) if len(sequence) == 6)
    probe = collate([validation[probe_index]])
    torch.manual_seed(experiment["seed"])
    rng = random.Random(experiment["seed"])
    model = Transformer(**config)
    for parameter in model.parameters():
        if parameter.ndim > 1:
            torch.nn.init.xavier_uniform_(parameter)
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment["learning_rate"],
                                 betas=tuple(experiment["betas"]))
    cap = experiment["generation_max_new_tokens"]
    snapshots, probabilities, logits_by_step, history = [], [], [], []
    record, probabilities0, logits0 = snapshot(model, probe, 0, cap)
    snapshots.append(record)
    probabilities.append(probabilities0)
    logits_by_step.append(logits0)
    history.append({"step": 0, "validation": evaluate(model, validation, generation_max_new_tokens=cap)})

    captured, raw_gradients = {}, {}

    def readout_hook(module, inputs, output):
        captured["first_logits"] = output.detach().clone()

    def gradient_hook(name):
        def observe(gradient):
            raw_gradients[name] = gradient.detach().clone()
        return observe

    before_bias = model.vocabulary_projection.bias.detach().clone()
    handles = [model.vocabulary_projection.register_forward_hook(readout_hook)]
    handles.extend(parameter.register_hook(gradient_hook(name))
                   for name, parameter in model.named_parameters())
    first_update = None
    arrays = {}
    for step in range(1, experiment["steps"] + 1):
        batch = collate(rng.sample(train, experiment["batch_size"]))
        loss = train_step(model, optimizer, batch)
        if step == 1:
            for handle in handles:
                handle.remove()
            valid = batch.target != PAD
            p = captured["first_logits"].softmax(-1)[valid].double().mean(0)
            frequencies = torch.bincount(batch.target[valid], minlength=len(VOCAB)).double() / valid.sum()
            expected_gradient = p - frequencies
            raw_bias_gradient = raw_gradients["vocabulary_projection.bias"].double()
            torch.testing.assert_close(expected_gradient, raw_bias_gradient, atol=4e-8, rtol=2e-6)
            raw_norm = torch.sqrt(sum(g.double().square().sum() for g in raw_gradients.values()))
            clipped_norm = torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters()))
            clipped_bias = model.vocabulary_projection.bias.grad.detach().double()
            after_bias = model.vocabulary_projection.bias.detach().clone()
            delta = (after_bias - before_bias).double()
            eps = optimizer.param_groups[0]["eps"]
            expected_delta = -experiment["learning_rate"] * clipped_bias / (clipped_bias.abs() + eps)
            torch.testing.assert_close(delta, expected_delta, atol=8e-9, rtol=2e-6)
            arrays.update(mean_probability=p.numpy(), reference_frequency=frequencies.numpy(),
                          raw_bias_gradient=raw_bias_gradient.numpy(),
                          clipped_bias_gradient=clipped_bias.numpy(), bias_delta=delta.numpy(),
                          first_source_ids=batch.source.numpy(), first_target_ids=batch.target.numpy(),
                          first_logits=captured["first_logits"].numpy())
            first_update = {"loss": loss, "evaluated_tokens": int(valid.sum()),
                            "raw_gradient_norm": float(raw_norm), "clipped_gradient_norm": float(clipped_norm),
                            "bias_gradient_max_error": float((expected_gradient-raw_bias_gradient).abs().max()),
                            "adam_bias_delta_max_error": float((delta-expected_delta).abs().max()),
                            "eos": {"mean_probability": float(p[EOS]), "reference_frequency": float(frequencies[EOS]),
                                    "raw_gradient": float(raw_bias_gradient[EOS]),
                                    "clipped_gradient": float(clipped_bias[EOS]), "bias_delta": float(delta[EOS])}}
            raw_gradients.clear()
        if step in (100, saved["step"]):
            record, observed_p, observed_logits = snapshot(model, probe, step, cap)
            snapshots.append(record)
            probabilities.append(observed_p)
            logits_by_step.append(observed_logits)
        if step % experiment["validation_every"] == 0 or step == experiment["steps"]:
            metrics = evaluate(model, validation, generation_max_new_tokens=cap)
            history.append({"step": step, "validation": metrics})
    for name, value in model.state_dict().items():
        if not torch.equal(value, reference.state_dict()[name]):
            raise AssertionError("replay differs from checkpoint; regenerate both in the same environment: " + name)
    selected = min(history, key=lambda row: row["validation"]["nll"])
    assert selected["step"] == saved["step"]
    arrays.update(snapshot_correct_probability=np.stack(probabilities),
                  snapshot_logits=np.stack(logits_by_step),
                  probe_source=probe.source[0].numpy(), probe_target=probe.target[0].numpy())
    report = {"checkpoint_sha256": sha256(checkpoint), "model_state_sha256": state_sha256(model),
              "replay_matches_checkpoint_exactly": True, "configuration": config, "experiment": experiment,
              "probe_selection": "first length-six validation source, selected before replay or inspecting predictions",
              "probe_validation_index": probe_index, "source": words(probe.source[0]),
              "reference": words(probe.target[0], include_special=True),
              "first_update": first_update, "snapshots": snapshots, "validation_history": history,
              "software": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                           for name in ("torch", "numpy", "matplotlib", "Pillow")}}}
    return reference.eval().requires_grad_(False), probe, report, arrays


@torch.no_grad()
def observe_cache(model, probe, cap):
    memory, allowed = encode_source(model, probe.source)
    cache = CachedDecoder(model, memory, allowed)
    static_pairs = [(k.clone(), v.clone()) for k, v in cache.cross_kv]
    prefix = probe.source.new_tensor([[BOS]])
    self_bytes, cross_bytes, errors, consumed, generated, logits_rows = [], [], [], [], [], []
    old_pairs = None
    for _ in range(cap):
        consumed.append(int(prefix[0, -1]))
        got = cache.step(prefix[:, -1:])
        full = decode_prefix(model, prefix, memory, allowed)[:, -1]
        torch.testing.assert_close(got, full, atol=1e-5, rtol=2e-6)
        errors.append(float((got-full).abs().max()))
        logits_rows.append(got[0].numpy().copy())
        if old_pairs is not None:
            for (k, v), (old_k, old_v) in zip(cache.self_kv, old_pairs):
                assert torch.equal(k[:, :, :-1], old_k) and torch.equal(v[:, :, :-1], old_v)
        for current, original in zip(cache.cross_kv, static_pairs):
            assert all(torch.equal(a, b) for a, b in zip(current, original))
        old_pairs = [(k.clone(), v.clone()) for k, v in cache.self_kv]
        self_bytes.append(sum(t.numel()*t.element_size() for pair in cache.self_kv for t in pair))
        cross_bytes.append(sum(t.numel()*t.element_size() for pair in cache.cross_kv for t in pair))
        choice = got.clone()
        choice[:, [PAD, BOS]] = -float("inf")
        token = int(choice.argmax(-1))
        full_choice = full.clone()
        full_choice[:, [PAD, BOS]] = -float("inf")
        assert token == int(full_choice.argmax(-1))
        generated.append(token)
        if token == EOS:
            break
        prefix = torch.cat((prefix, prefix.new_tensor([[token]])), dim=1)
    assert EOS not in consumed and generated == probe.target[0].tolist()
    full_logits, weights = model(probe.source, prefix, return_weights=True)
    np.testing.assert_allclose(np.stack(logits_rows), full_logits[0].numpy(), atol=1e-5, rtol=2e-6)
    maps = weights["decoder"][0][0].numpy()
    np.testing.assert_allclose(maps.sum(-1), 1, atol=2e-7)
    assert np.count_nonzero(np.triu(maps, 1)) == 0
    width, layers = model.d_model, len(model.decoder_layers)
    element_size = next(model.parameters()).element_size()
    assert self_bytes == [2*layers*t*width*element_size for t in range(1, len(consumed)+1)]
    assert cross_bytes == [2*layers*probe.source.shape[1]*width*element_size]*len(consumed)
    return {"consumed": [VOCAB[t] for t in consumed], "generated": [VOCAB[t] for t in generated],
            "self_kv_bytes": self_bytes, "cross_kv_bytes": cross_bytes,
            "maximum_logit_difference": max(errors), "logit_differences": errors,
            "old_self_kv_unchanged": True, "static_cross_kv_unchanged": True}, {
                "decoder_self_maps": maps, "cached_logits": np.stack(logits_rows),
                "self_kv_bytes": np.array(self_bytes), "cross_kv_bytes": np.array(cross_bytes)}


def figures(report, arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    paths = []
    labels = [word.strip("<>").upper() if i < 3 else word for i, word in enumerate(VOCAB)]
    x = np.arange(len(VOCAB))
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.0), layout="constrained", sharex=True)
    axes[0].bar(x-.18, arrays["mean_probability"], .34, color="#2468a0", label="Mean predicted probability")
    axes[0].bar(x+.18, arrays["reference_frequency"], .34, color="#c8772f", label="Reference-label frequency")
    axes[0].set(title="(a) Grade all nonpadding target positions in the first minibatch", ylabel="Probability / frequency")
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    axes[1].bar(x-.18, arrays["raw_bias_gradient"], .34, color="#2468a0", label="Before clipping")
    axes[1].bar(x+.18, arrays["clipped_bias_gradient"], .34, color="#c8772f", label="Sent to Adam")
    axes[1].set(title="(b) Readout-bias gradient = mean probability − label frequency", ylabel="Signed gradient")
    axes[1].legend(frameon=False, fontsize=9, ncol=2)
    axes[2].bar(x, 1000*arrays["bias_delta"], .62, color="#168477")
    axes[2].set(title="(c) First Adam step changes the nine vocabulary biases", ylabel="Bias change × 1000", xlabel="Vocabulary entry")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(-1.3, 1.3)
    for ax in axes:
        ax.axhline(0, color="#555555", lw=.7)
    path = output / "ch06_real_update.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig = plt.figure(figsize=(8.4, 5.5), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=(1.15, 1))
    ax = fig.add_subplot(grid[0])
    data = arrays["snapshot_correct_probability"]
    im = ax.imshow(data, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    targets = [VOCAB[t].strip("<>") for t in arrays["probe_target"]]
    ax.set_xticks(range(data.shape[1]), [f"{i+1}: {word}" for i, word in enumerate(targets)])
    ax.set_yticks(range(3), [f"Update {row['step']}" for row in report["snapshots"]])
    for (r, c), value in np.ndenumerate(data):
        ax.text(c, r, f"{value:.3f}", ha="center", va="center", color="white" if value>.65 else "#222222", fontsize=9)
    ax.set(title="(a) Probability of each correct label, given its reference prefix", xlabel="Prediction position and reference label")
    fig.colorbar(im, ax=ax, shrink=.75, label="Correct-label probability")
    ax = fig.add_subplot(grid[1])
    ax.axis("off")
    rows = [[str(r["step"]), f"{r['correct_next_tokens']}/{r['target_tokens']}",
             r["generated"].replace("<eos>", "EOS")] for r in report["snapshots"]]
    table = ax.table(cellText=rows, colLabels=["Update", "Reference-prefix\ncorrect choices", "Free generation (own prefix)"],
                     colWidths=[.11, .24, .65], cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.1)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#e7eef4")
        if col == 2:
            cell.get_text().set_fontsize(8)
    ax.set_title("(b) Supplied reference prefixes and self-generated prefixes are different tests", loc="left")
    fig.suptitle("Validation source: " + report["source"], fontsize=12)
    path = output / "ch06_learning_snapshots.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.1), layout="constrained")
    data = arrays["decoder_self_maps"][0]
    names = [f"{i+1}: {word.strip('<>')}" for i, word in enumerate(report["cache"]["consumed"])]
    im = axes[0].imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(names)), names, rotation=20)
    axes[0].set_yticks(range(len(names)), names)
    for (r, c), value in np.ndenumerate(data):
        axes[0].text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if value>.65 else "#222222")
    axes[0].set(title="(a) Measured self-attention on the generated prefix: layer 1, head 1",
                xlabel="Consumed decoder input (key)", ylabel="Decoder input (query)")
    fig.colorbar(im, ax=axes[0], shrink=.8)
    t = np.arange(1, len(names)+1)
    axes[1].plot(t, arrays["self_kv_bytes"]/1024, "o-", color="#2468a0", label="Self-KV: grows with consumed inputs")
    axes[1].plot(t, arrays["cross_kv_bytes"]/1024, "s-", color="#168477", label="Cross-KV: fixed source")
    axes[1].set(title="(b) Actual KV tensor storage across both decoder layers",
                xlabel="Consumed decoder inputs (EOS is selected, then generation stops)", ylabel="Stored KV (KiB)", xticks=t)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    path = output / "ch06_real_cache.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return {path.name: sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch06-figures"))
    args = parser.parse_args()
    model, probe, report, arrays = replay(args.checkpoint)
    cache, cache_arrays = observe_cache(model, probe, report["experiment"]["generation_max_new_tokens"])
    report["cache"] = cache
    arrays.update(cache_arrays)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "observations.npz"
    np.savez_compressed(archive, **arrays)
    report["artifacts"] = {archive.name: sha256(archive), **figures(report, arrays, args.output_dir)}
    root = Path(__file__).resolve().parents[2]
    paths = (Path(__file__).resolve(), Path(__file__).with_name("training.py"),
             Path(__file__).with_name("data.py"), Path(__file__).with_name("decoding.py"),
             root / "examples/ch05_transformer/model.py")
    report["source_sha256"] = {str(path.relative_to(root)): sha256(path) for path in paths}
    (args.output_dir / "TRAINED_FIGURES_RUN.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
