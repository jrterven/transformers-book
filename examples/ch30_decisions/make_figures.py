"""Draw the Chapter 30 figures from recorded decisions and simulator frames."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RECORDS = Path(__file__).with_name("records")
BLUE, ORANGE, INK = "#23638e", "#b35331", "#223a4f"
COLORS = (BLUE, ORANGE)
STYLES = ("-", "--")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#b7c7d3", "axes.axisbelow": True,
    "xtick.color": INK, "ytick.color": INK, "grid.color": "#dce6ee",
    "pdf.fonttype": 42,
})


def save(figure, output, filename):
    figure.savefig(output / filename, pad_inches=0.035, bbox_inches="tight",
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def plot_reacher(episodes, output):
    fig, (path, distance) = plt.subplots(
        1, 2, figsize=(8.25, 3.15), gridspec_kw={"width_ratios": (1, 1.35)}, layout="constrained",
    )
    chosen = [row for row in episodes if row["seed"] == 30]
    chosen.sort(key=lambda row: not row["history"])
    for record, color, style, label in zip(
        chosen, COLORS, STYLES, ("With history", "Current observation only"), strict=True,
    ):
        target = 100 * np.asarray(record["target"])
        tip = 100 * np.asarray(record["tip_xy"])
        path.plot(*tip.T, color=color, linestyle=style, linewidth=1.8)
        path.plot(*tip[-1], marker="o", color=color, markersize=4)
        distance.plot(np.arange(len(tip)) * record["period_s"],
                      np.linalg.norm(tip - target, axis=1), color=color,
                      linewidth=1.8, linestyle=style, label=label)
    path.plot(*tip[0], marker="o", markerfacecolor="white", markeredgecolor=INK, markersize=5)
    path.plot(*target, marker="*", color=INK, markersize=9)
    path.annotate("Start", tip[0], xytext=(-5, -8), textcoords="offset points",
                  ha="right", va="top", fontsize=9)
    path.annotate("Target", target, xytext=(0, -10), textcoords="offset points",
                  ha="center", va="top", fontsize=9)
    path.set(title="(a) Tip paths from the same reset", xlabel="x (cm)", ylabel="y (cm)")
    path.set_aspect("equal", adjustable="box")
    path.margins(0.19)
    distance.set(title="(b) Distance to the target", xlabel="Simulated time (s)", ylabel="Distance (cm)",
                 xlim=(0, 1))
    for ax in (path, distance):
        ax.grid(True, linewidth=0.7)
    fig.legend(*distance.get_legend_handles_labels(), loc="outside lower center",
               ncols=2, frameon=False, fontsize=10)
    save(fig, output, "ch30_vintix_reacher.pdf")


def plot_frames(source, output):
    fig, axes = plt.subplots(2, 4, figsize=(8.2, 4.05), layout="constrained")
    samples = ((0, "Start"), (46, "Grasp"), (64, "Lift"), (76, "Goal reached"))
    for j, (step, phase) in enumerate(samples):
        for i, camera in enumerate(("image", "image2")):
            ax = axes[i, j]
            ax.imshow(plt.imread(source / f"{step:03d}-{camera}.png"))
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"{phase} · {step}", color=INK, fontsize=11, pad=5)
            if j == 0:
                ax.text(-0.07, 0.5, ("Table view", "Wrist view")[i],
                        transform=ax.transAxes, rotation=90, va="center", ha="center",
                        fontsize=10)
    save(fig, output, "ch30_smolvla_frames.pdf")


def plot_bowl(episodes, output):
    fig, (distance, height) = plt.subplots(1, 2, figsize=(8.25, 3.25), layout="constrained")
    selected = sorted((row for row in episodes if row["initial_state"] == 2),
                      key=lambda row: row["horizon"])
    for row, color, style in zip(selected, COLORS, STYLES, strict=True):
        bowl = np.array([o["bowl"] for o in row["trace"]])
        plate = np.array([o["plate"] for o in row["trace"]])
        gap = 100 * np.linalg.norm((bowl - plate)[:, :2], axis=1)
        rise = 100 * (bowl[:, 2] - bowl[0, 2])
        queries = [q["step"] for q in row["queries"]]
        label = rf"$K={row['horizon']}$"
        for ax, values in ((distance, gap), (height, rise)):
            ax.plot(values, color=color, linewidth=1.8, linestyle=style, label=label)
            ax.plot(queries, values[queries], "o", color=color, markersize=3,
                    markerfacecolor="white", markeredgewidth=0.8)
            ax.plot(len(values) - 1, values[-1], color=color, markersize=6,
                    marker="D" if row["success"] else "X")
        endpoint = len(rise) - 1
        label = f"Success at {endpoint}" if row["success"] else "Time limit"
        offset = (12, 30) if row["success"] else (-8, 24)
        height.annotate(label, (endpoint, rise[-1]), xytext=offset, textcoords="offset points",
                        ha="left" if row["success"] else "right", color=color, fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": color})
    distance.set(title="(a) Horizontal bowl-to-plate gap", ylabel="Distance (cm)")
    height.set(title="(b) Bowl height above its start", ylabel="Height change (cm)")
    last_step = max(len(row["actions"]) for row in selected)
    for ax in (distance, height):
        ax.set(xlabel="Actions executed", xlim=(-4, last_step + 6))
        ax.grid(True, linewidth=0.7)
    fig.legend(*distance.get_legend_handles_labels(), loc="outside lower center",
               ncols=2, frameon=False, fontsize=10)
    save(fig, output, "ch30_smolvla_refresh.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vintix", type=Path, default=RECORDS / "vintix_reacher.json")
    parser.add_argument("--smolvla", type=Path, default=RECORDS / "smolvla_libero.json")
    parser.add_argument("--frames", type=Path, default=RECORDS / "frames")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vintix = json.loads(args.vintix.read_text(encoding="utf-8"))
    smolvla = json.loads(args.smolvla.read_text(encoding="utf-8"))
    plot_reacher(vintix["episodes"], args.output_dir)
    plot_frames(args.frames, args.output_dir)
    plot_bowl(smolvla["episodes"], args.output_dir)


if __name__ == "__main__":
    main()
