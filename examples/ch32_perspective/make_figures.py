"""Draw the Chapter 32 model figures from the recorded Qwen run."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .run_pretrained import MODES


RECORD = Path(__file__).with_name("records") / "qwen_history.json"
BLUE, GREEN, ORANGE = "#24658b", "#40836e", "#ad502b"
INK, MUTED, GRID = "#203748", "#506677", "#dbe6ed"
MIB = 1024 ** 2

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#bccbd5",
    "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "pdf.fonttype": 42,
})


def save(figure, output, filename):
    figure.savefig(output / filename, pad_inches=0.05, bbox_inches="tight",
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def plot_history(cache, output):
    fig, (total, parts) = plt.subplots(1, 2, figsize=(8.3, 3.15), layout="constrained")
    tokens = [row["tokens"] for row in cache]
    recurrent = [(row["state"] + row["conv"]) / MIB for row in cache]
    kv = [row["kv"] / MIB for row in cache]
    totals = [a + b for a, b in zip(recurrent, kv, strict=True)]

    total.set_title("(a) Total persistent history", loc="left")
    total.plot(tokens, totals, color=INK, linewidth=1.8, marker="o",
               markerfacecolor="white", markeredgewidth=1.6)
    total.fill_between(tokens, totals, color=GRID, alpha=0.55)
    total.annotate(f"{totals[0]:.2f} MiB", (tokens[0], totals[0]),
                   xytext=(3, 17), textcoords="offset points", fontsize=10)
    total.annotate(f"{totals[-1]:.2f} MiB", (tokens[-1], totals[-1]),
                   xytext=(0, 9), textcoords="offset points", fontsize=10, ha="right")

    parts.set_title("(b) What the model keeps", loc="left")
    parts.plot(tokens, recurrent, color=GREEN, linewidth=1.8, marker="s", markersize=4)
    parts.plot(tokens, kv, color=BLUE, linewidth=1.8, marker="o", markersize=4)
    parts.annotate("Recurrent state + convolution\n19.6875 MiB",
                   (tokens[-1], recurrent[-1]), xytext=(0, 8), textcoords="offset points",
                   ha="right", fontsize=10, color=GREEN)
    parts.annotate("Attention keys and values\n+24 KiB per token",
                   (tokens[-1], kv[-1]), xytext=(0, 8), textcoords="offset points",
                   ha="right", fontsize=10, color=BLUE)

    for ax in (total, parts):
        ax.set(xlabel="Prefix length (tokens)", ylabel="History (MiB)",
               xlim=(0, tokens[-1] + 22), xticks=[0, 128, 256, 384, 512],
               ylim=(0, 36), yticks=[0, 8, 16, 24, 32])
        ax.grid(axis="y", linewidth=0.75)
    save(fig, output, "ch32_hybrid_history.pdf")


def plot_choices(report, output):
    fig, (choices, counts) = plt.subplots(
        1, 2, figsize=(8.45, 3.35), gridspec_kw={"width_ratios": (1, 1.15)}, layout="constrained",
    )
    rows = (3, 2, 1, 0)
    choice_labels = ("Full log", "Recent log", "Moved register", "Selected entry")
    examples = [case for case in report["cases"]
                if case["codes"] == ["Cedar", "Maple"] and case["active"] == "A"]
    choices.set_title("(a) Swapping the two codes", loc="left", pad=33)
    choices.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
    for case, offset, marker, color in zip(examples, (0.14, -0.14), ("o", "D"), (BLUE, ORANGE), strict=True):
        probabilities = [case["scores"][mode]["probabilities"][0] for mode in MODES]
        choices.plot(probabilities, [row + offset for row in rows], linestyle="", marker=marker,
                     markersize=6, color=color, label=f"Target {case['answer']}")
    choices.set(xlim=(0, 1), ylim=(-0.48, 3.58),
                xticks=(0, 0.25, 0.5, 0.75, 1), xlabel="Cedar's share within the two candidates")
    choices.set_yticks(rows, choice_labels)
    choices.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    choices.grid(axis="x", linewidth=0.65)
    choices.grid(axis="y", linewidth=0.65)
    choices.legend(loc="lower left", bbox_to_anchor=(-0.015, 1.002), frameon=False,
                   ncols=2, fontsize=9.5, handletextpad=0.3, columnspacing=0.7, borderaxespad=0)

    counts.set_title("(b) Correct choices and pairs", loc="left", pad=33)
    counts.set(xlim=(0, 3), ylim=choices.get_ylim())
    counts.set_axis_off()
    columns = (
        ("Individual\nchoices", "correct", "total"),
        ("Assignment\nswaps", "registry_pairs", "registry_total"),
        ("Selection\nflips", "selection_pairs", "selection_total"),
    )
    for x, (label, count_key, total_key) in enumerate(columns):
        counts.text(x + 0.5, 3.59, label, ha="center", va="bottom", fontsize=10)
        for y, mode in zip(rows, MODES, strict=True):
            row = report["scores"][mode]
            correct, size = row[count_key], row[total_key]
            color = "#dceee8" if correct == size else "#eff4f7"
            counts.add_patch(plt.Rectangle((x + 0.045, y - 0.31), 0.91, 0.62, color=color))
            counts.text(x + 0.5, y, f"{correct}/{size}", ha="center", va="center",
                        fontweight="bold" if correct else "normal", color=INK if correct else MUTED)
    save(fig, output, "ch32_paired_choices.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.record.read_text(encoding="utf-8"))
    plot_history(report["cache"], args.output_dir)
    plot_choices(report, args.output_dir)


if __name__ == "__main__":
    main()
