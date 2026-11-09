"""Plot observed states, gates, and predictions of the chapter's trained LSTM.

Reproduce the checkpoint with run_examples --train --steps 750 first.
The frozen model performs the color-reversal task, not language translation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from .data import BOS, EOS, VOCAB, collate, make_splits, words
from .model import RecurrentSeq2Seq, greedy_decode
from .run_examples import evaluate


STATE_SOURCES = ((3, 4, 5), (6, 4, 5))
FAILURE_SOURCE = (4, 5, 8, 3, 3, 3)
DISPLAY_COORDINATES = tuple(range(8))
GATE_COORDINATE = 0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def trace_lstm(layer, embeddings):
    """Expose every gate and state for one forward, one-layer LSTM sequence.

    PyTorch stores stacked gate blocks in input, forget, candidate, output order.
    Every observed state is checked against the independent native LSTM kernel.
    """
    if layer.num_layers != 1 or layer.bidirectional or layer.proj_size:
        raise ValueError("expected an unprojected, one-layer forward LSTM")
    if embeddings.ndim != 2 or not len(embeddings):
        raise ValueError("expected a nonempty [time, embedding] input")
    hidden = embeddings.new_zeros(layer.hidden_size)
    cell = torch.zeros_like(hidden)
    observed = {key: [] for key in ("forget", "write", "candidate", "expose",
                                    "old_cell", "retained", "written", "cell", "hidden")}
    for x in embeddings:
        raw = (x @ layer.weight_ih_l0.T + hidden @ layer.weight_hh_l0.T
               + layer.bias_ih_l0 + layer.bias_hh_l0)
        write_raw, forget_raw, candidate_raw, expose_raw = raw.chunk(4)
        write, forget = torch.sigmoid(write_raw), torch.sigmoid(forget_raw)
        candidate, expose = torch.tanh(candidate_raw), torch.sigmoid(expose_raw)
        old_cell = cell
        retained, written = forget * old_cell, write * candidate
        cell = retained + written
        hidden = expose * torch.tanh(cell)
        values = (forget, write, candidate, expose, old_cell,
                  retained, written, cell, hidden)
        for key, value in zip(observed, values):
            observed[key].append(value.detach().clone())
    actual = {k: torch.stack(v) for k, v in observed.items()}
    native_hidden, (_, native_cell) = layer(embeddings[None])
    torch.testing.assert_close(actual["hidden"], native_hidden[0], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(actual["cell"][-1], native_cell[0, 0], atol=3e-6, rtol=3e-6)
    # Verify every intermediate cell against an independently evaluated prefix.
    for t in range(len(embeddings)):
        _, (_, c) = layer(embeddings[None, :t + 1])
        torch.testing.assert_close(actual["cell"][t], c[0, 0], atol=3e-6, rtol=3e-6)
    return {k: v.cpu().numpy() for k, v in actual.items()}


@torch.no_grad()
def observations(checkpoint):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = RecurrentSeq2Seq().eval().requires_grad_(False)
    model.load_state_dict(saved["model"])
    train, validation, test = make_splits(seed=saved["configuration"]["seed"])
    assert FAILURE_SOURCE in test and FAILURE_SOURCE not in train
    traces = [trace_lstm(model.encoder, model.source_embedding(torch.tensor(s)))
              for s in STATE_SOURCES]
    for a, b in zip(STATE_SOURCES[0][1:], STATE_SOURCES[1][1:]):
        assert a == b
    assert not np.allclose(traces[0]["hidden"][-1], traces[1]["hidden"][-1])
    failure_trace = trace_lstm(model.encoder, model.source_embedding(torch.tensor(FAILURE_SOURCE)))
    batch = collate([FAILURE_SOURCE])
    reference = batch.target[0].tolist()
    teacher_logits = model(batch.source, batch.lengths, batch.decoder_input)[0]
    generated = greedy_decode(model, batch.source, batch.lengths, 7)[0].tolist()
    # This selected held-out example ends at the same step on both paths.
    assert len(generated) == len(reference) and generated[-1] == EOS
    assert generated != reference
    generated_input = torch.tensor([[BOS] + generated[:-1]])
    self_logits = model(batch.source, batch.lengths, generated_input)[0]
    assert self_logits.argmax(-1).tolist() == generated
    teacher_input = batch.decoder_input[0].tolist()
    first_input_difference = next(i for i, (a, b) in enumerate(
        zip(teacher_input, generated_input[0].tolist())) if a != b)
    torch.testing.assert_close(teacher_logits[:first_input_difference],
                               self_logits[:first_input_difference], atol=0, rtol=0)
    probabilities = torch.softmax(torch.stack((teacher_logits, self_logits)).double(), -1).numpy()
    np.testing.assert_allclose(probabilities.sum(-1), 1, atol=1e-12, rtol=0)
    parameter_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        parameter_hash.update(name.encode())
        parameter_hash.update(value.numpy().tobytes())
    coord = GATE_COORDINATE
    gate_example = {k: float(v[1, coord]) for k, v in failure_trace.items()}
    gate_example.update(source_position=2, input_token=VOCAB[FAILURE_SOURCE[1]],
                        coordinate=coord + 1)
    report = dict(
        task="color reversal; locally trained encoder-decoder; no attention",
        mode="frozen weights; eval; CPU float32; no sampling",
        checkpoint_sha256=sha256(checkpoint), parameter_tensors_sha256=parameter_hash.hexdigest(),
        selected_update=saved["selected_step"], configuration=saved["configuration"],
        vocabulary=list(VOCAB), state_width=model.encoder.hidden_size,
        state_sources=[dict(ids=list(s), text=words(s),
                            split="train" if s in train else "validation" if s in validation
                                  else "test" if s in test else "illustration outside all splits")
                       for s in STATE_SOURCES],
        displayed_coordinates=[i + 1 for i in DISPLAY_COORDINATES],
        gate_coordinate=coord + 1, gate_source=list(FAILURE_SOURCE), gate_example=gate_example,
        failure=dict(test_index=test.index(FAILURE_SOURCE), source=list(FAILURE_SOURCE),
                     reference=reference, teacher_inputs=teacher_input,
                     teacher_choices=teacher_logits.argmax(-1).tolist(), generated=generated,
                     generated_inputs=generated_input[0].tolist(),
                     first_different_input_step=first_input_difference + 1,
                     reference_token_probabilities=[float(probabilities[0, t, token])
                                                   for t, token in enumerate(reference)],
                     generated_reference_probabilities=[float(probabilities[1, t, token])
                                                       for t, token in enumerate(reference)]),
        test_metrics=evaluate(model, test),
        software={name: importlib.metadata.version(name) for name in ("torch", "numpy", "matplotlib", "Pillow")},
    )
    arrays = {f"state_{j}_{k}": value for j, trace in enumerate(traces) for k, value in trace.items()}
    arrays.update({f"gate_{k}": value for k, value in failure_trace.items()})
    arrays.update(probabilities=probabilities, teacher_logits=teacher_logits.numpy(),
                  generated_logits=self_logits.numpy())
    return report, arrays


def figures(report, arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    paths = []
    cols = list(DISPLAY_COORDINATES)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.9), constrained_layout=True)
    for row, variable in enumerate(("hidden", "cell")):
        scale = max(float(np.abs(arrays[f"state_{j}_{variable}"][:, cols]).max()) for j in (0, 1))
        for j in (0, 1):
            ax = axes[row, j]
            values = arrays[f"state_{j}_{variable}"][:, cols]
            im = ax.imshow(values, cmap="PuOr_r", vmin=-scale, vmax=scale, aspect="auto")
            ax.set_yticks(range(3), [f"{i+1}: {VOCAB[t]}" for i, t in enumerate(STATE_SOURCES[j])])
            ax.set_xticks(range(len(cols)), [i + 1 for i in cols])
            ax.set_title(("Hidden state h" if row == 0 else "Cell state c") + ("; first color red" if j == 0 else "; first color yellow"), fontsize=10)
            if row == 1:
                ax.set_xlabel("Coordinate (first 8 of 64)")
            for (r, c), value in np.ndenumerate(values):
                ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(value) > .65 * scale else "#202020")
        fig.colorbar(im, ax=list(axes[row]), shrink=.85, label="Coordinate value")
    path = output / "ch02_trained_states.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.8), constrained_layout=True)
    steps = np.arange(1, len(FAILURE_SOURCE) + 1)
    for key, label, color in (("forget", "Retain gate f", "#267C98"),
                              ("write", "Input gate i", "#AF5482"),
                              ("expose", "Output gate o", "#6B8C52")):
        axes[0].plot(steps, arrays[f"gate_{key}"][:, GATE_COORDINATE], "o-", label=label, color=color)
    axes[0].set(ylim=(-.03, 1.08), ylabel="Gate multiplier", title="(a) Learned gates for coordinate 1")
    axes[0].legend(ncols=3, loc="lower right", fontsize=9)
    retained = arrays["gate_retained"][:, GATE_COORDINATE]
    written = arrays["gate_written"][:, GATE_COORDINATE]
    axes[1].bar(steps - .15, retained, width=.28, color="#267C98", label="Retained: f × old c")
    axes[1].bar(steps + .15, written, width=.28, color="#AF5482", label="Written: i × g")
    axes[1].plot(steps, retained + written, "D-", color="#27374B", label="New c = retained + written")
    axes[1].axhline(0, color="#555555", lw=.6)
    axes[1].set(ylabel="Cell contribution", title="(b) Two contributions, one updated cell")
    axes[1].legend(ncols=1, loc="best", fontsize=9)
    for ax in axes:
        ax.set_xticks(steps, [f"{i}: {VOCAB[t]}" for i, t in zip(steps, FAILURE_SOURCE)])
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    path = output / "ch02_trained_gates.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4), constrained_layout=True)
    failure = report["failure"]
    for j, (ax, inputs, title) in enumerate(zip(axes,
            (failure["teacher_inputs"], failure["generated_inputs"]),
            ("(a) Correct preceding tokens", "(b) Model's preceding choices"))):
        values = arrays["probabilities"][j]
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(inputs)), [f"{t + 1}: {VOCAB[k]}" for t, k in enumerate(inputs)])
        ax.set_xticks(range(len(VOCAB)), [s.replace("<", "").replace(">", "") for s in VOCAB], rotation=50, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Candidate next token (all 9 entries)")
        ax.set_ylabel("Step: decoder input")
        for t, probs in enumerate(values):
            chosen = int(probs.argmax())
            ax.add_patch(Rectangle((chosen - .48, t - .47), .96, .94, fill=False, edgecolor="#E49A3A", lw=1.5))
            for token, probability in enumerate(probs):
                if probability >= .1:
                    ax.text(token, t, f"{probability:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if probability > .6 else "#202020")
    fig.colorbar(im, ax=list(axes), shrink=.9, label="Full-vocabulary probability")
    path = output / "ch02_trained_predictions.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, arrays = observations(args.checkpoint)
    data_path = args.output_dir / "ch02_trained_observations.npz"
    np.savez_compressed(data_path, **arrays)
    images = figures(report, arrays, args.output_dir)
    report["source_sha256"] = sha256(__file__)
    report["model_source_sha256"] = sha256(Path(__file__).with_name("model.py"))
    report["artifacts"] = {p.name: sha256(p) for p in [data_path, *images]}
    (args.output_dir / "TRAINED_FIGURES_RUN.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
