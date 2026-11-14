"""Draw the chapter figures from the published model run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np


BLUE, ORANGE, TEAL, INK, GRID = "#236594", "#b45422", "#08776c", "#183247", "#d7e1e9"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
}


def intervention_diagram():
    fig, ax = plt.subplots(figsize=(8.4, 2.85), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 3.0))
    ax.axis("off")
    xs = (1.25, 3.75, 6.25, 8.75)

    def box(x, y, label, accent):
        ax.add_patch(FancyBboxPatch((x-1.08, y-.38), 2.16, .76,
                                   boxstyle="round,pad=0.03,rounding_size=0.09",
                                   facecolor="white", edgecolor=accent, linewidth=1.6))
        ax.text(x, y, label, ha="center", va="center", fontsize=10)

    def arrow(start, end, color=INK, width=1.25):
        ax.annotate("", xy=end, xytext=start, arrowprops={
            "arrowstyle": "->", "color": color, "linewidth": width, "shrinkA": 0, "shrinkB": 0,
        })

    ax.text(.15, 2.90, "Donor run", va="center", weight="bold", color=ORANGE)
    ax.text(.15, 1.26, "Patched recipient run", va="center", weight="bold", color=BLUE)
    for y, texts, color in (
        (2.30, ("Replace my\nmailing address", "Run blocks\n1 through $\\ell$",
                r"Save the vector $\mathbf{h}^{\mathrm{B}}_{\ell,i}$", "Donor score\n(reference)"), ORANGE),
        (.66, ("Show my\nmailing address", "Run blocks\n1 through $\\ell$",
               "Replace only\nposition $i$", "Run remaining blocks\nand read the score"), BLUE),
    ):
        for x, label in zip(xs, texts, strict=True):
            box(x, y, label, color)
        for left, right in zip(xs[:-1], xs[1:], strict=True):
            arrow((left+1.10, y), (right-1.10, y), color)
    arrow((xs[2], 1.90), (xs[2], 1.06), TEAL, 1.8)
    ax.text(xs[2]+.18, 1.46, "Copy one activation", va="center", color=TEAL, fontsize=10)
    return fig


def patch_map(report):
    pair = report["pairs"][0]
    series = ("read_recipient", "write_recipient")
    matrices = [np.array([[v["shift"] for v in block[key]] for block in pair["patches"]])
                for key in series]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.1), sharey=True, layout="constrained")
    scale = max(abs(v).max() for v in matrices)
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)
    labels = ("Before\nverb", "Verb", "After\nverb", "Answer", "Colon")
    for ax, values, title in zip(axes, matrices,
                                 ("(a) Patch into the read request", "(b) Patch into the update request"),
                                 strict=True):
        im = ax.imshow(values, norm=norm, cmap="RdBu_r", aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks([0, 3, 7, 11, 15, 19, 23], [1, 4, 8, 12, 16, 20, 24])
        ax.set(title=title, xlabel="Position in the prompt")
        for edge in ("left", "bottom"):
            ax.spines[edge].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(values), 1), minor=True)
        ax.grid(which="minor", color="#48647a", alpha=.17, linewidth=.45)
        ax.tick_params(which="minor", length=0)
    axes[0].set_ylabel("Block whose output is patched")
    colorbar = fig.colorbar(im, ax=axes, shrink=.85, pad=.035)
    colorbar.set_label("Change in $z_B-z_A$ (nats)", fontsize=10)
    return fig


def routing_tradeoff(report):
    groups = report["routing"].values()
    policies = ("Constrained", "Defer below 0.8")
    fig, ax = plt.subplots(figsize=(8.4, 3.65), layout="constrained")
    xs = np.arange(4)
    for j, policy in enumerate(policies):
        summary = [g["metrics"][policy] for g in groups]
        clear_defers = sum(row["target"] != "C" and row["predictions"][policy] == "C"
                           for g in groups for row in g["cases"])
        actual_write = [row for g in groups for row in g["cases"] if row["target"] == "B"]
        values = (sum(x["correct"] for x in summary), sum(x["right_to_clarify"] for x in summary),
                  clear_defers, sum(row["predictions"][policy] == "B" for row in actual_write))
        denominators = (18, 6, 12, 6)
        bars = ax.bar(xs+(j-.5)*.34, np.array(values)/denominators, .31,
                      label=policy, color=(BLUE, TEAL)[j], hatch=("", "//")[j],
                      edgecolor=("#19496c", "#095e56")[j], linewidth=.6)
        ax.bar_label(bars, [f"{v}/{n}" for v,n in zip(values, denominators, strict=True)],
                     padding=3, fontsize=10)
    ax.set_xticks(xs, ("Correct actions", "Clarified when\nneeded", "Deferred a\nclear request",
                       "Correctly routed\nan update"))
    ax.set(ylim=(0, 1.17), ylabel="Fraction of relevant requests")
    ax.set_yticks([0, .25, .5, .75, 1], ["0", ".25", ".50", ".75", "1"])
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.00))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.record.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for name, figure in (("patch_workflow", intervention_diagram()),
                             ("patch_map", patch_map(report)),
                             ("routing_tradeoff", routing_tradeoff(report))):
            figure.savefig(args.output / f"ch18_{name}.pdf")
            plt.close(figure)


if __name__ == "__main__":
    main()
