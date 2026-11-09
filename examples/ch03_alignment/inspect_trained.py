"""Expose actual scores, masked reads, and decoder timing in a trained model.

The two interventions hold the encoder memory and preceding decoder state
fixed. They are local diagnostics, not new greedy-evaluation results.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch

from examples.ch02_recurrence.data import BOS, EOS, PAD, VOCAB, collate, make_splits
from .model import AttentiveSeq2Seq, greedy_decode
from .run_examples import alignment_example, evaluate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def open_read(model, state, memory):
    """Reconstruct a read, checking it against the model's native operation."""
    annotations, projected, valid = memory
    features = torch.tanh(projected + model.attention.query(state[0])[:, None, :]
                          + model.attention.bias)
    raw_scores = model.attention.energy(features).squeeze(-1)
    weights = raw_scores.masked_fill(~valid, -torch.inf).softmax(-1)
    contributions = weights[:, :, None] * annotations
    context = contributions.sum(1)
    native_context, native_weights = model.attention(
        state[0], annotations, projected, valid)
    torch.testing.assert_close(weights, native_weights)
    torch.testing.assert_close(context, native_context, atol=2e-7, rtol=2e-6)
    lower = annotations.masked_fill(~valid[:, :, None], torch.inf).amin(1)
    upper = annotations.masked_fill(~valid[:, :, None], -torch.inf).amax(1)
    assert bool(((context >= lower - 2e-7) & (context <= upper + 2e-7)).all())
    return raw_scores, weights, contributions, context


@torch.no_grad()
def prefix_state(model, source, prefix):
    """Return memory/state after consuming exactly the supplied token prefix."""
    batch = collate([tuple(source)])
    memory, state = model.encode(batch.source, batch.lengths)
    for token in prefix:
        _, state, _ = model.step(torch.tensor([token]), state, memory)
    return memory, state


@torch.no_grad()
def observations(checkpoint):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = saved["configuration"]
    model = AttentiveSeq2Seq(embedding_width=cfg["embedding_width"],
                            hidden_width=cfg["hidden_width"],
                            alignment_width=cfg["alignment_width"])
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    _, _, test = make_splits(cfg["seed"])
    evaluated = evaluate(model, test)
    trace = alignment_example(model, test, evaluated["predictions"])
    if not trace["ended_in_eos"] or trace["generated"] != trace["target"]:
        raise ValueError("These figures require the selected correct example ending in EOS")
    source, generated = trace["source"], trace["generated"]
    # EOS is the final prediction. Its input is the final generated color.
    memory, state = prefix_state(model, source, [BOS] + generated[:-2])
    raw, weights, contributions, context = open_read(model, state, memory)
    logits, _, native = model.step(torch.tensor([generated[-2]]), state, memory)
    torch.testing.assert_close(native[0], torch.tensor(trace["attention"][-1]))
    arrays = {
        "source_ids": np.array(source),
        "eos_query": state[0][0].numpy(),
        "eos_annotations": memory[0][0].numpy(),
        "eos_scores": raw[0].numpy(),
        "eos_weights": weights[0].numpy(),
        "eos_contributions": contributions[0].numpy(),
        "eos_context": context[0].numpy(),
        "eos_vocabulary_probabilities": logits.softmax(-1)[0].numpy(),
    }

    # First test source of length three; selection does not inspect the effect.
    short_index = next(i for i, sequence in enumerate(test) if len(sequence) == 3)
    short = test[short_index]
    short_batch = collate([short])
    short_tokens, _ = greedy_decode(model, short_batch.source, short_batch.lengths, 7)
    short_generated = short_tokens[0].tolist()
    if short_generated != list(reversed(short)) + [EOS]:
        raise ValueError("The padding illustration requires a correct short example")
    padded = torch.nn.functional.pad(short_batch.source, (0, 3), value=PAD)
    pad_memory, pad_state = model.encode(padded, short_batch.lengths)
    for token in [BOS] + short_generated[:-2]:
        _, pad_state, _ = model.step(torch.tensor([token]), pad_state, pad_memory)
    pad_raw, pad_weights, _, pad_context = open_read(model, pad_state, pad_memory)
    unmasked_weights = pad_raw.softmax(-1)
    unmasked_context = (unmasked_weights[:, :, None] * pad_memory[0]).sum(1)
    previous = torch.tensor([short_generated[-2]])
    correct_logits, _, _ = model.step(previous, pad_state, pad_memory)
    joined = torch.cat((model.target_embedding(previous), unmasked_context), -1)
    changed_hidden, _ = model.decoder(joined, pad_state)
    changed_probabilities = model.output(changed_hidden).softmax(-1)
    padding_mass = unmasked_weights[0, 3:].sum()
    # Zero PAD annotations imply a precisely scaled read in this intervention.
    torch.testing.assert_close(unmasked_context, (1 - padding_mass) * pad_context,
                               atol=2e-7, rtol=2e-6)
    assert torch.equal(pad_memory[0][0, 3:], torch.zeros_like(pad_memory[0][0, 3:]))
    arrays.update({
        "padding_source_ids": padded[0].numpy(),
        "padding_scores": pad_raw[0].numpy(),
        "padding_correct_weights": pad_weights[0].numpy(),
        "padding_unmasked_weights": unmasked_weights[0].numpy(),
        "padding_correct_context": pad_context[0].numpy(),
        "padding_unmasked_context": unmasked_context[0].numpy(),
        "padding_correct_probabilities": correct_logits.softmax(-1)[0].numpy(),
        "padding_unmasked_probabilities": changed_probabilities[0].numpy(),
    })

    # Step 2: replace the previous-token input, preserving both preceding states.
    # Step 3: feed the SAME next input to both branches, isolating the one change.
    timing_memory, timing_state = prefix_state(model, source, [BOS])
    original, replacement = generated[0], VOCAB.index("blue")
    if original == replacement:
        raise ValueError("Timing illustration requires a changed input token")
    branch = []
    for token in (original, replacement):
        logits2, state2, attention2 = model.step(torch.tensor([token]), timing_state,
                                                timing_memory)
        _, _, attention3 = model.step(torch.tensor([generated[1]]), state2, timing_memory)
        branch.append((logits2.softmax(-1), attention2, attention3))
    assert torch.equal(branch[0][1], branch[1][1])
    arrays.update({
        "timing_current_probabilities": torch.cat([b[0] for b in branch]).numpy(),
        "timing_current_weights": torch.cat([b[1] for b in branch]).numpy(),
        "timing_next_weights": torch.cat([b[2] for b in branch]).numpy(),
    })
    parameter_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        parameter_hash.update(name.encode())
        parameter_hash.update(value.numpy().tobytes())
    report = {
        "checkpoint_sha256": sha256(checkpoint),
        "parameter_sha256": parameter_hash.hexdigest(),
        "selected_update": saved["selected_step"], "configuration": cfg,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__,
                        "matplotlib": importlib.metadata.version("matplotlib"),
                        "pillow": importlib.metadata.version("Pillow"),
                        "device": "cpu", "threads": 1},
        "test_token_nll": evaluated["token_nll"],
        "test_exact_matches": evaluated["exact_matches"],
        "vocabulary": list(VOCAB),
        "read": {"test_index": trace["test_index"], "source": [VOCAB[i] for i in source],
                 "generated": [VOCAB[i] for i in generated], "step": len(generated),
                 "selection": trace["selection"], "scores": arrays["eos_scores"].tolist(),
                 "weights": arrays["eos_weights"].tolist(),
                 "context_first_two": arrays["eos_context"][:2].tolist(),
                 "p_eos": float(logits.softmax(-1)[0, EOS])},
        "padding": {"test_index": short_index, "source": [VOCAB[i] for i in short],
                    "selection": "first length-three test source, before inspecting effect",
                    "step": len(short_generated), "added_pad_positions": 3,
                    "unmasked_padding_mass": float(padding_mass),
                    "correct_p_eos": float(correct_logits.softmax(-1)[0, EOS]),
                    "unmasked_p_eos": float(changed_probabilities[0, EOS]),
                    "control": "memory, parameters, query, cell state and token fixed; mask removed for one read"},
        "timing": {"step": 2, "original_input": VOCAB[original],
                   "changed_input": VOCAB[replacement], "common_next_input": VOCAB[generated[1]],
                   "current_max_attention_difference": float((branch[0][1] - branch[1][1]).abs().max()),
                   "next_max_attention_difference": float((branch[0][2] - branch[1][2]).abs().max()),
                   "current_max_probability_difference": float((branch[0][0] - branch[1][0]).abs().max()),
                   "control": "same encoded memory and preceding hidden/cell; one input changed, next input held fixed"},
    }
    return report, arrays


def figures(report, arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    labels = [f"{i + 1}\n{word}" for i, word in enumerate(report["read"]["source"])]
    blue, orange, teal = "#2468a0", "#ce762b", "#168477"

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.6), layout="constrained")
    x = np.arange(len(labels))
    for ax, values, title, ylabel in (
            (axes[0], arrays["eos_scores"], "(a) Scores before softmax", "Score"),
            (axes[1], arrays["eos_weights"], "(b) Weights over six source positions", "Weight")):
        ax.bar(x, values, color=blue, width=.62)
        ax.set_xticks(x, labels)
        ax.set(title=title, ylabel=ylabel)
        ax.axhline(0, color="#555555", lw=.7)
        ax.margins(y=.22)
        for j, value in enumerate(values):
            ax.annotate(f"{value:.3f}", (j, value), xytext=(0, 5 if value >= 0 else -14),
                        textcoords="offset points", ha="center", fontsize=10)
    axes[1].set_ylim(0, 1)
    parts = arrays["eos_contributions"][:, :2]
    values = np.vstack((parts, arrays["eos_context"][:2]))
    x = np.arange(7)
    axes[2].bar(x - .18, values[:, 0], .34, color=teal, label="Feature 1")
    axes[2].bar(x + .18, values[:, 1], .34, color=orange, label="Feature 2")
    axes[2].set_xticks(x, labels + ["SUM\ncontext"])
    axes[2].set(title="(c) Weighted annotation coordinates, then their sum", ylabel="Signed feature value")
    axes[2].axhline(0, color="#555555", lw=.7)
    axes[2].axvline(5.5, color="#999999", ls=":")
    axes[2].legend(loc="best", ncol=2, frameon=False)
    fig.suptitle("A real read: the EOS prediction at step 7", fontsize=14)
    path = output / "ch03_trained_read.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.0), layout="constrained")
    words = [VOCAB[int(i)].replace("<pad>", "PAD") for i in arrays["padding_source_ids"]]
    x = np.arange(6)
    axes[0].bar(x - .18, arrays["padding_correct_weights"], .34, color=blue, label="Correct source mask")
    axes[0].bar(x + .18, arrays["padding_unmasked_weights"], .34, color=orange, label="Mask removed for this read")
    axes[0].set_xticks(x, [f"{j+1}\n{word}" for j, word in enumerate(words)])
    axes[0].set_ylim(0, 1)
    axes[0].axvspan(2.5, 5.5, color="#ececec", zorder=0)
    axes[0].set(title="(a) Empty positions still compete if their scores are allowed", ylabel="Attention weight")
    axes[0].legend(frameon=False, fontsize=10)
    for j in range(3, 6):
        axes[0].annotate(f"{arrays['padding_unmasked_weights'][j]:.3f}",
                         (j + .18, arrays["padding_unmasked_weights"][j]),
                         xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    x = np.arange(8)
    axes[1].bar(x - .18, arrays["padding_correct_context"][:8], .34, color=blue)
    axes[1].bar(x + .18, arrays["padding_unmasked_context"][:8], .34, color=orange)
    axes[1].set_xticks(x, [str(j+1) for j in x])
    axes[1].axhline(0, color="#555555", lw=.7)
    axes[1].set(title="(b) Zero PAD annotations dilute the read", xlabel="Context feature (first 8 of 64)", ylabel="Signed feature value")
    fig.suptitle("One controlled change: remove the source score mask", fontsize=14)
    path = output / "ch03_trained_padding.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.0), layout="constrained")
    x = np.arange(6)
    names = ("Input red", "Input blue (changed)")
    for ax, key, title in (
            (axes[0], "timing_current_weights", "(a) Step 2: attention is calculated before reading that input"),
            (axes[2], "timing_next_weights", "(c) Step 3: both branches receive white, but their queries differ")):
        ax.bar(x - .18, arrays[key][0], .34, color=blue, label=names[0])
        ax.bar(x + .18, arrays[key][1], .34, color=orange, label=names[1])
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set(title=title, ylabel="Attention weight")
    axes[0].legend(frameon=False, ncol=2, fontsize=10)
    probs = arrays["timing_current_probabilities"]
    x = np.arange(len(VOCAB))
    axes[1].bar(x - .18, probs[0], .34, color=blue)
    axes[1].bar(x + .18, probs[1], .34, color=orange)
    axes[1].set_xticks(x, [word.strip("<>").upper() if word.startswith("<") else word for word in VOCAB], fontsize=10)
    axes[1].set_ylim(0, 1)
    axes[1].set(title="(b) Step 2: vocabulary probabilities can already change", ylabel="Token probability")
    fig.suptitle("Replace one previous-token input: red with blue", fontsize=14)
    path = output / "ch03_trained_timing.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return {path.name: sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch03-inspection"))
    args = parser.parse_args()
    report, arrays = observations(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "observations.npz"
    np.savez_compressed(archive, **arrays)
    report["arrays_sha256"] = sha256(archive)
    report["figure_sha256"] = figures(report, arrays, args.output_dir)
    report["source_sha256"] = {
        str(path.relative_to(Path.cwd())): sha256(path)
        for path in (Path(__file__).resolve(), Path(__file__).with_name("model.py").resolve(),
                     Path(__file__).with_name("run_examples.py").resolve())}
    manifest = args.output_dir / "TRAINED_FIGURES_RUN.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("read", "padding", "timing")}, indent=2))
    print("Saved three measured figures, complete arrays, and provenance in", args.output_dir)


if __name__ == "__main__":
    main()
