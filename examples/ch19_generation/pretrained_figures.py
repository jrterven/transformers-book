"""Draw Chapter 19's figures from saved translation and summarization runs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np


RECORD = Path(__file__).with_name("PRETRAINED_RUN.json")
AUDIT = Path(__file__).with_name("SUMMARY_FACT_AUDIT.json")
INK = "#253548"
BLUE = "#256497"
GREEN = "#217960"
ORANGE = "#b85b25"
GRAY = "#68798a"
GRID = "#dfe6ec"


def save(fig, destination, name):
    fig.savefig(destination / f"ch19_pretrained_{name}.png", dpi=210, facecolor="white")
    plt.close(fig)


def token_labels(pieces):
    return ["EOS" if piece == "</s>" else piece.removeprefix("▁") for piece in pieces]


def translation_trace(case, destination):
    out = case["outputs"]["beam_raw"]
    src = token_labels(case["source"]["pieces"])
    tgt = token_labels(out["pieces"])
    weights = np.array(out["cross_attention"])
    fig, (heat, probs) = plt.subplots(
        1, 2, figsize=(8.35, 3.5), sharey=True, layout="constrained", width_ratios=(1.65, 1))
    image = heat.imshow(weights, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row, values in enumerate(weights):
        for column, value in enumerate(values):
            heat.text(column, row, f"{value:.2f}", va="center", ha="center",
                      color="white" if value > .65 else INK, fontsize=10)
    for src_word, tgt_word in (("bus", "autobús"), ("blue", "azul")):
        heat.add_patch(Rectangle((src.index(src_word) - .5, tgt.index(tgt_word) - .5), 1, 1,
                                 edgecolor=ORANGE, lw=2, fill=False))
    heat.set(title="(a) Cross-attention", xlabel="English source token",
             ylabel="Spanish token predicted at this step",
             yticks=range(len(tgt)), yticklabels=tgt)
    heat.set_xticks(range(len(src)), src, rotation=25, ha="right")
    heat.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=heat, fraction=.038, pad=.02, ticks=(0, .5, 1))
    colorbar.ax.tick_params(labelsize=9)
    values = np.exp(out["token_log_p"])
    probs.barh(range(len(tgt)), values, color=GREEN, height=.6)
    for row, value in enumerate(values):
        probs.text(value - .025, row, f"{value:.1%}", va="center", ha="right",
                   color="white", fontweight="bold", fontsize=10)
    probs.set(title="(b) Selected token", xlabel="Next-token probability",
              xlim=(0, 1), xticks=(0, .5, 1))
    probs.xaxis.set_major_formatter(PercentFormatter(1.0))
    probs.tick_params(axis="y", left=False)
    probs.spines["left"].set_visible(False)
    save(fig, destination, "translation_trace")


def target_loss(case, destination):
    targets = case["supplied_targets"]
    first, second = (token_labels(row["pieces"]) for row in targets)
    labels = [a if a == b else f"{a} / {b}" for a, b in zip(first, second, strict=True)]
    fig, (steps, means) = plt.subplots(
        1, 2, figsize=(8.35, 2.9), layout="constrained", width_ratios=(1.75, 1))
    positions = np.arange(len(labels))
    for i, (row, label, color) in enumerate(zip(targets, ("Blue bus", "Red bus"), (BLUE, ORANGE))):
        losses = -np.array(row["token_log_p"])
        steps.bar(positions + (i - .5) * .35, losses, width=.35, color=color, label=label)
        average = losses.mean()
        means.barh(i, average, height=.45, color=color)
        means.text(average + .035, i, f"{average:.3f}", va="center", fontsize=11, color=INK)
    steps.set(title="(a) Loss at each label", ylabel=r"Negative log-probability $-\ln p_t$",
              ylim=(0, 7.1), yticks=(0, 2, 4, 6))
    steps.set_xticks(positions, labels, rotation=21, ha="right")
    steps.legend(loc="upper right", frameon=False, fontsize=10)
    means.set(title="(b) Mean over seven labels", xlabel="Mean token loss",
              yticks=(0, 1), yticklabels=("Blue bus", "Red bus"), xlim=(0, 1.27))
    means.invert_yaxis()
    save(fig, destination, "target_loss")


def beam_paths(case, destination):
    paths = [case["outputs"][k] for k in ("greedy", "beam_raw")]
    g, b = paths
    fork = next(i for i, pair in enumerate(zip(g["ids"], b["ids"])) if pair[0] != pair[1])
    labels, colors = ("Greedy", "Beam width 4"), (ORANGE, BLUE)
    fig, (local, total) = plt.subplots(
        1, 2, figsize=(8.35, 3.05), layout="constrained", width_ratios=(1, 1.55))
    for i, (out, label, color) in enumerate(zip(paths, labels, colors)):
        token = token_labels(out["pieces"])[fork]
        probability = np.exp(out["token_log_p"][fork])
        local.barh(i, probability, color=color, height=.5)
        local.text(probability - .015, i, f"{probability:.3f}", ha="right", va="center",
                   color="white", fontweight="bold")
        steps = np.arange(fork, out["length"] + 1)
        scores = np.r_[0., np.cumsum(out["token_log_p"])][fork:]
        total.plot(steps, scores, marker="o", ms=3.5, lw=2, color=color,
                   label=f"{label}  ({out['sum_log_p']:.2f})")
        total.scatter(steps[-1], scores[-1], marker="D", s=42, color=color, zorder=3)
        local.text(.012, i, token, color="white", va="center", fontweight="bold")
    local.set(title=f"(a) First different choice, step {fork + 1}",
              xlabel="Probability after the shared prefix", xlim=(0, .5),
              xticks=(0, .25, .5), yticks=(0, 1), yticklabels=labels)
    local.invert_yaxis()
    total.set(title="(b) Scores through the end of each path", xlabel="Generated tokens",
              ylabel="Cumulative log-probability", xticks=range(fork, max(g["length"], b["length"]) + 1),
              ylim=(-6.62, -2.54))
    total.grid(axis="y", color=GRID)
    total.legend(loc="upper right", frameon=False, fontsize=9.5)
    save(fig, destination, "beam_paths")


def summary_rows(summary, audit):
    for row in audit["outputs"]:
        source = summary["sources"][row["source"]]
        out = source["outputs"][str(row["cap"])]
        provided = " ".join(summary["report_sentences"][:row["sentences_available"]])
        if out["text"] != row["text"] or provided != source["source"]["text"]:
            raise ValueError("The manual audit does not match the recorded source and output")
        yield row, source["source"], out


def summary_budget(rows, destination):
    labels = [f"{'Full' if row['source'] == 'full' else 'Lead'} / {row['cap']}" for row, _, _ in rows]
    fig, (enc, dec) = plt.subplots(
        1, 2, figsize=(8.35, 2.95), layout="constrained", sharey=True, width_ratios=(1, 1.25))
    for i, (row, src, out) in enumerate(rows):
        size = len(src["ids"])
        enc.barh(i, size, color=BLUE, height=.51)
        enc.text(size - 3, i, str(size), va="center", ha="right", color="white", fontweight="bold")
        color = GREEN if out["ended"] else ORANGE
        end = out["length"]
        dec.barh(i, end, color=color, height=.51)
        dec.plot([row["cap"], row["cap"]], [i - .37, i + .37], color=INK, lw=1.2)
        status = "EOS" if out["ended"] else "cap"
        dec.text(end + 2.1, i, f"{end} · {status}", va="center", color=INK, fontsize=10)
    enc.set(title="(a) Tokens sent to the encoder", xlabel="Input length", xlim=(0, 94),
            xticks=(0, 30, 60, 90), yticks=range(len(labels)), yticklabels=labels)
    enc.invert_yaxis()
    dec.set(title="(b) Tokens emitted by the decoder", xlabel="Output length and limit",
            xlim=(0, 78), xticks=(0, 18, 36, 54, 72))
    dec.tick_params(axis="y", left=False)
    dec.spines["left"].set_visible(False)
    save(fig, destination, "summary_budget")


def summary_audit(rows, facts, destination):
    fig, (cells, overlap) = plt.subplots(
        1, 2, figsize=(8.35, 3.05), layout="constrained", width_ratios=(1.68, 1))
    values = []
    for fact in facts:
        values.append([
            2 if fact["id"] in row["conveyed"] else
            1 if fact["sentence"] <= row["sentences_available"] else 0 for row, _, _ in rows
        ])
    cmap = ListedColormap(["#eef1f4", "#fae4d4", "#d6eee4"])
    cells.imshow(values, cmap=cmap, vmin=-.5, vmax=2.5, aspect="auto")
    for i, states in enumerate(values):
        for j, state in enumerate(states):
            word = ("Absent", "Omitted", "Kept")[state]
            cells.text(j, i, word, ha="center", va="center", fontsize=10, color=INK)
    labels = [f"{'Full' if row['source'] == 'full' else 'Lead'}\n{row['cap']} max" for row, _, _ in rows]
    cells.set(title="(a) Required facts", xticks=range(len(rows)), xticklabels=labels,
              yticks=range(len(facts)), yticklabels=[fact["label"] for fact in facts])
    cells.set_xticks(np.arange(len(rows)) - .5, minor=True)
    cells.set_yticks(np.arange(len(facts)) - .5, minor=True)
    cells.grid(which="minor", color="white", lw=1.5)
    cells.tick_params(which="both", length=0)
    for spine in cells.spines.values():
        spine.set_visible(False)
    for i, (row, _, out) in enumerate(rows):
        score = out["overlap"]["f1"]
        overlap.barh(i, score, color=BLUE if row["source"] == "full" else GRAY, height=.54)
        overlap.text(score + .018, i, f"{score:.3f}", va="center", color=INK, fontsize=10)
    overlap.set(title="(b) ROUGE-L", xlabel=r"$F_1$ against one reference", xlim=(0, .55),
                xticks=(0, .2, .4), yticks=range(len(rows)),
                yticklabels=[label.replace("\n", " / ").removesuffix(" max") for label in labels])
    overlap.invert_yaxis()
    save(fig, destination, "summary_audit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch19-figures"))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10.8, "axes.labelcolor": INK, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    record = json.loads(args.record.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {row["id"]: row for row in record["translation"]["cases"]}
    translation_trace(cases["blue_bus"], args.output_dir)
    target_loss(cases["blue_bus"], args.output_dir)
    beam_paths(cases["order"], args.output_dir)
    rows = list(summary_rows(record["summary"], audit))
    summary_budget(rows, args.output_dir)
    summary_audit(rows, audit["facts"], args.output_dir)


if __name__ == "__main__":
    main()
