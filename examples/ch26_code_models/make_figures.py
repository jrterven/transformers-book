"""Draw the endpoint example and plots from Chapter 26's recorded model outputs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


INK = "#21394e"
BLUE = "#3f6d98"
GREEN = "#22684f"
ORANGE = "#985220"
GRAY = "#4d6678"
PALE_GREEN = "#e1eee7"
PALE_ORANGE = "#faeadb"
PALE_GRAY = "#eef0f1"
CHAT_MODELS = {"qwen38": "Qwen3.8-27B", "gemma26": "Gemma 4 26B", "liquid24": "LFM2.5-24B-A2B"}
REPAIR_MODELS = {"qwen38": "Qwen3.8", "gemma26": "Gemma 4"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def draw_sweep(output):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 3.30), sharex=True,
                                      gridspec_kw={"height_ratios": [1.2, 1]})
    top.axvline(2, linestyle="--", color="#a8b9c6", lw=1.1)
    for y, start, end, color, name in ((2, 0, 2, BLUE, "A"), (1, 2, 4, GREEN, "B")):
        top.plot([start, end], [y, y], color=color, linewidth=5, solid_capstyle="butt")
        top.scatter([start, end], [y, y], s=48, edgecolors=color,
                    facecolors=[color, "white"], linewidth=1.3, zorder=4)
        top.text(start-0.25, y, name, ha="right", va="center", weight="bold")
    top.scatter([2], [0], s=48, edgecolors=ORANGE, facecolors="white", linewidth=1.3, zorder=4)
    top.text(2.13, 0, "Empty booking [2, 2)", va="center", color=ORANGE)
    top.text(2.10, 2.05, "at time 2:  −1 + 1 + 1 − 1 = 0", va="bottom", color=GRAY)
    top.set_ylim(-0.4, 2.67)
    top.set_yticks([])
    top.spines[:].set_visible(False)
    top.tick_params(axis="x", length=0)
    bottom.axvline(2, linestyle="--", color="#a8b9c6", lw=1.1)
    bottom.step([-0.4, 0, 2, 4, 4.45], [0, 1, 1, 0, 0], where="post", color=BLUE, linewidth=2.1)
    bottom.scatter([0, 2, 4], [1, 1, 0], s=24, color=BLUE, zorder=4)
    bottom.set(xlim=(-0.55, 4.55), ylim=(-0.17, 1.35), xticks=range(5), yticks=[0, 1],
               ylabel="Active bookings", xlabel="Integer time")
    bottom.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(hspace=0.07)
    save(fig, output / "ch26_booking_sweep.pdf")


def draw_fim(record, output):
    traces = record["results"]["operator_traces"]
    conditions = ["no_suffix", "half_open", "closed"]
    fig, ax = plt.subplots(figsize=(7.3, 2.45))
    for row, name in enumerate(conditions):
        values = traces[name]["probabilities"]
        for symbol, shift, color in (("<", -0.16, GREEN), ("<=", 0.16, BLUE)):
            value = values[symbol]["probability"]
            ax.barh(row + shift, value, height=0.27, color=color)
            ax.text(value + 0.013, row + shift, f"{100*value:.1f}%", va="center", fontsize=10)
    ax.set(xlim=(0, 1.04), xticks=[0, .25, .5, .75, 1], yticks=[0, 1, 2],
           yticklabels=["Empty suffix", "Half-open assertions", "Closed assertions"],
           xlabel="Next-token probability over the full vocabulary")
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dce3ea", linewidth=0.8)
    ax.legend(handles=[Patch(color=GREEN, label="<"), Patch(color=BLUE, label="<=")],
              ncol=2, loc="lower right", bbox_to_anchor=(1, 1.01), frameon=False)
    save(fig, output / "ch26_fim_measured.pdf")


def result_cell(ax, column, row, text, passed):
    if passed is True:
        fill, ink = PALE_GREEN, GREEN
    elif passed is False:
        fill, ink = PALE_ORANGE, ORANGE
    else:
        fill, ink = PALE_GRAY, GRAY
    ax.add_patch(Rectangle((column-0.44, row-0.32), 0.88, 0.64, facecolor=fill,
                           edgecolor="white", linewidth=1.2))
    ax.text(column, row, text, ha="center", va="center", color=ink, weight="bold")


def result_axes(ax, columns, rows):
    ax.set(xlim=(-.51, len(columns)-.49), ylim=(len(rows)-.54, -.49),
           xticks=range(len(columns)), xticklabels=columns, yticks=range(len(rows)), yticklabels=rows)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=4)
    ax.spines[:].set_visible(False)


def draw_generation(records, output):
    fig, ax = plt.subplots(figsize=(8.3, 2.2))
    seeds = [sample["seed"] for sample in records["qwen38"]["results"]["generation"]["samples"]]
    for row, key in enumerate(CHAT_MODELS):
        generation = records[key]["results"]["generation"]
        if [sample["seed"] for sample in generation["samples"]] != seeds:
            raise ValueError(f"Generation seeds differ for {key}")
        for column, sample in enumerate(generation["samples"]):
            evaluation = sample["evaluation"]
            status = evaluation["status"]
            if sample["finish"] != "stop":
                value = "Limit"
            elif status == "tested":
                value = f"{sum(item['passed'] for item in evaluation['checks'])}/{generation['case_count']}"
            else:
                value = status.capitalize()
            result_cell(ax, column, row, value, evaluation["passed"])
    result_axes(ax, [str(seed) for seed in seeds], list(CHAT_MODELS.values()))
    save(fig, output / "ch26_generation_attempts.pdf")


def draw_repair(records, output):
    fig, ax = plt.subplots(figsize=(8.7, 2.9))
    base = records["qwen38"]["results"]["repair"]["baseline"]
    evaluations = [("Before repair", base)]
    for key, name in REPAIR_MODELS.items():
        repair = records[key]["results"]["repair"]
        evaluations.extend([(f"{name} · issue", repair["issue_only"]["evaluation"]),
                            (f"{name} · + helper", repair["with_policy"]["evaluation"])])
    for row, (_, evaluation) in enumerate(evaluations):
        by_name = {check["case"]: check["passed"] for check in evaluation["checks"]}
        for column, base_check in enumerate(base["checks"]):
            passed = by_name.get(base_check["case"])
            value = {True: "P", False: "F", None: "—"}[passed]
            result_cell(ax, column, row, value, passed)
    cases = ["Touching", "Empty", "Cleanup", "Cancelled", "Interior", "One unit", "Separate", "Negative"]
    result_axes(ax, cases, [name for name, _ in evaluations])
    save(fig, output / "ch26_repair_checks.pdf")


def export_examples(fim, measured, destination):
    destination.mkdir(parents=True, exist_ok=True)
    half_open = fim["results"]["completions"][0]
    qwen = measured["qwen38"]["results"]
    sample = qwen["generation"]["samples"][0]
    repair = qwen["repair"]["with_policy"]
    if not sample["evaluation"]["passed"] or not repair["evaluation"]["passed"]:
        raise ValueError("The selected chapter examples must pass their recorded suites")
    examples = {
        "overlaps_fim.txt": half_open["raw"],
        "peak_bookings.py": sample["candidate"],
        "conflicts_with_policy.py": repair["candidate"],
    }
    for name, source in examples.items():
        (destination / name).write_text(source.rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--examples-output", type=Path, help="also export the three recorded LaTeX listings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fim = json.loads((args.records / "fim15.json").read_text())
    draw_sweep(args.output)
    draw_fim(fim, args.output)
    measured = {key: json.loads((args.records / f"{key}.json").read_text()) for key in CHAT_MODELS}
    draw_generation(measured, args.output)
    draw_repair(measured, args.output)
    if args.examples_output:
        export_examples(fim, measured, args.examples_output)
