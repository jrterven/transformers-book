"""Draw the four measured figures in Chapter 28 from saved predictions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter
import numpy as np

from .tabular_data import load_tables


RECORD = Path(__file__).with_name("records") / "tabular_runs.json"
INK, GRID, MUTED = "#223a4f", "#dde5ea", "#526777"
MODELS = {
    "linear": ("Linear", "#55636e", "--", "s"),
    "trees": ("Boosted trees", "#26725a", "-.", "^"),
    "tabicl": ("TabICLv2", "#246596", "-", "o"),
    "mitra": ("Mitra-v2", "#a64e2a", ":", "D"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42,
})


def save(fig, directory, filename):
    fig.savefig(
        directory / filename, bbox_inches="tight", pad_inches=0.07,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def grid(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def full_context(dataset):
    return dataset["runs"][str(len(dataset["training_ids"]))]


def draw_classification(raisin, output):
    from sklearn.metrics import roc_curve

    actual = np.array(raisin["actual"])
    full = full_context(raisin)
    fig, axes = plt.subplots(1, 2, figsize=(8.45, 3.60), layout="constrained")
    roc, reliability = axes
    for ax in axes:
        grid(ax)
        ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=0.9, alpha=0.85)
        ax.set(xlim=(-0.025, 1.025), ylim=(-0.025, 1.025))
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    for method in ("linear", "tabicl", "mitra"):
        name, color, style, marker = MODELS[method]
        probability = np.asarray(full[method]["point"])
        fpr, tpr, _ = roc_curve(actual, probability)
        auc = full[method]["metrics"]["auc"]
        roc.plot(fpr, tpr, color=color, linestyle=style, linewidth=1.75,
                 label=f"{name}  {auc:.3f}")
        hit = (probability >= 0.5) & (actual == 1)
        false_alarm = (probability >= 0.5) & (actual == 0)
        roc.plot(false_alarm.sum() / (actual == 0).sum(), hit.sum() / (actual == 1).sum(),
                 marker=marker, color=color, markersize=4.5)
    roc.set(title="(a) Ranking across thresholds", xlabel="False-positive rate",
            ylabel="True-positive rate")
    roc.legend(loc="lower right", frameon=True, title="Model and ROC AUC", fontsize=9.2)

    probability = np.asarray(full["tabicl"]["point"])
    edges = np.linspace(0, 1, 6)
    bins = np.minimum(np.searchsorted(edges, probability, side="right") - 1, 4)
    count_offsets = ((6, 4), (3, -13), (7, 3), (-15, 5), (-29, 1))
    for i, offset in enumerate(count_offsets):
        selected = bins == i
        count = selected.sum()
        if not count:
            continue
        mean_probability = probability[selected].mean()
        observed_rate = actual[selected].mean()
        color = MODELS["tabicl"][1]
        reliability.scatter(mean_probability, observed_rate, s=27, color=color, zorder=4)
        reliability.annotate(f"n={count}", (mean_probability, observed_rate),
                             xytext=offset, textcoords="offset points", fontsize=9)
    reliability.set(title="(b) Reliability of TabICLv2", xlabel="Mean predicted probability",
                    ylabel="Observed fraction of Besni")
    save(fig, output, "ch28_raisin_results.pdf")


def draw_regression(concrete, output):
    actual = np.array(concrete["actual"])
    full = full_context(concrete)
    fig, axes = plt.subplots(1, 2, figsize=(8.10, 3.80), sharey=True, layout="constrained")
    for i, (ax, method) in enumerate(zip(axes, ("tabicl", "mitra"), strict=True)):
        name, color, _, marker = MODELS[method]
        prediction = full[method]["point"]
        ax.plot([0, 85], [0, 85], "--", linewidth=1, color=MUTED, zorder=1)
        ax.scatter(actual, prediction, s=19, color=color, alpha=0.55,
                   edgecolors="none", marker=marker, zorder=2)
        grid(ax)
        score = full[method]["metrics"]["mae_mpa"]
        ax.text(0.05, 0.93, f"MAE {score:.2f} MPa", transform=ax.transAxes,
                va="top", fontsize=10.5, color=color)
        ax.set(title=f"({chr(97+i)}) {name}", xlabel="Measured strength (MPa)",
               xlim=(0, 85), ylim=(0, 85), xticks=(0, 20, 40, 60, 80))
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Predicted strength (MPa)")
    save(fig, output, "ch28_concrete_results.pdf")


def draw_recipe(concrete, table, output):
    test = np.searchsorted(table.row_ids, concrete["test_ids"])
    if not np.array_equal(table.row_ids[test], concrete["test_ids"]):
        raise ValueError("the recipe plot must use the recorded test rows")
    ages = table.x[test, -1]
    groups = table.groups[test]
    # Select by the number of observed ages; ties favor the smallest group ID.
    group = max(np.unique(groups), key=lambda g: len(np.unique(ages[groups == g])))
    selected = np.flatnonzero(groups == group)
    selected = selected[np.argsort(ages[selected], kind="stable")]
    observed = np.asarray(concrete["actual"])[selected]
    full = full_context(concrete)
    fig, axes = plt.subplots(
        1, 2, figsize=(8.45, 3.40), sharey=True, sharex=True, layout="constrained"
    )
    for i, (ax, method) in enumerate(zip(axes, ("tabicl", "mitra"), strict=True)):
        name, color, _, marker = MODELS[method]
        days = ages[selected]
        prediction = np.asarray(full[method]["point"])[selected]
        low, _, high = np.asarray(full[method]["quantiles"])[selected].T
        ax.vlines(days, low, high, linewidth=2.2, alpha=0.64, color=color, zorder=1,
                  label="10th–90th percentiles")
        ax.plot(days, prediction, marker=marker, markersize=4.5, linewidth=1.2,
                color=color, label="Predicted mean", zorder=2)
        ax.scatter(days, observed, marker="x", s=35, linewidths=1.45, color=INK,
                   label="Measured strength", zorder=3)
        ax.set(title=f"({chr(97+i)}) {name}", xlabel="Curing age (days, log scale)", ylim=(8, 77))
        ax.set_xscale("log")
        ticks = np.unique(days)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([str(int(day)) for day in ticks], rotation=25, ha="right")
        ax.xaxis.set_minor_locator(NullLocator())
        grid(ax, "y")
    axes[0].set_ylabel("Strength (MPa)")
    axes[1].legend(loc="lower right", fontsize=9, frameon=True)
    save(fig, output, "ch28_concrete_recipe.pdf")


def draw_context_sizes(tables, output):
    fig, axes = plt.subplots(1, 2, figsize=(8.45, 3.40), layout="constrained")
    configs = (
        ("raisin", "brier", "(a) Raisin classification", "Binary Brier score"),
        ("concrete", "mae_mpa", "(b) Concrete regression", "Mean absolute error (MPa)"),
    )
    for ax, (table, metric, title, ylabel) in zip(axes, configs, strict=True):
        runs = tables[table]["runs"]
        sizes = sorted(int(n) for n in runs)
        for method, (name, color, style, marker) in MODELS.items():
            values = [runs[str(n)][method]["metrics"][metric] for n in sizes]
            ax.plot(sizes, values, label=name, color=color, linestyle=style,
                    marker=marker, linewidth=1.8, markersize=5)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set(title=title, xlabel="Labeled rows supplied", ylabel=ylabel)
        grid(ax, "y")
    axes[1].legend(loc="upper right", bbox_to_anchor=(1, 0.77), fontsize=9.5, frameon=True)
    save(fig, output, "ch28_context_sizes.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=RECORD)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = json.loads(args.predictions.read_text())["datasets"]
    draw_classification(datasets["raisin"], args.output_dir)
    draw_regression(datasets["concrete"], args.output_dir)
    draw_recipe(datasets["concrete"], load_tables()["concrete"], args.output_dir)
    draw_context_sizes(datasets, args.output_dir)


if __name__ == "__main__":
    main()
