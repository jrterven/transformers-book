"""Plot measured evaluations from the saved SmolLM2 runs for Chapters 16 and 17."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..ch16_adaptation.real_data import LABELS
from .context_evaluation import POSITIONS, RECORD_COUNTS, REPORTS
from .model import binary_metrics, reliability_bins


HERE = Path(__file__).resolve().parent
ADAPTATION_RUN = HERE.parent / "ch16_adaptation/PRETRAINED_FIGURES_RUN.json"
CONTEXT_RUN = HERE / "CONTEXT_RUN.json"
STAGES = ("base", "sft", "dpo")
NAMES = ("Published base", "After SFT", "After DPO")
EDGES = np.linspace(0, 1, 6)


def resampling_distribution(differences):
    """Exact distribution of the mean of iid draws from these observed units."""
    values = np.asarray(differences, dtype=int)
    count = len(values)
    weights = np.bincount(values + 1, minlength=3) / count
    mass = np.array([1.0])
    for _ in range(count):
        mass = np.convolve(mass, weights)
    support = np.arange(-count, count + 1) / count
    ends = support[np.searchsorted(mass.cumsum(), [0.025, 0.975])]
    return {"unit_count": count, "support": support.tolist(), "mass": mass.tolist(),
            "central_quantiles": ends.tolist(), "standard_deviation": float(values.std() / np.sqrt(count))}


def adaptation_evaluation(record):
    held = record["held"]
    truth = np.array([row["label"] for row in held])
    families = np.array([row["family"] for row in held])
    ids = [f"{row['family']}/{row['account']}" for row in held]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate held-out message ID")
    stages = {}
    for stage in STAGES:
        data = record["stages"][stage]
        correct = np.asarray(data["correct_logp"])
        alternative = np.asarray(data["other_label_logp"])
        lookup = np.where(truth == 0, correct, alternative)
        update = np.where(truth == 1, correct, alternative)
        total = np.logaddexp(lookup, update)
        p_update = np.exp(update - total)
        candidate = (update > lookup).astype(int)
        outputs = []
        for row_id, y, prediction, p, log_l, log_u, m, generated in zip(
                ids, truth, candidate, p_update, lookup, update, np.exp(total),
                data["generated"], strict=True):
            answer = generated["text"].strip()
            outcome = "invalid"
            if answer in LABELS:
                outcome = "correct" if answer == LABELS[y] else "wrong label"
            if (outcome == "correct") != generated["exact_label"]:
                raise ValueError(f"generation does not match the recorded label at {row_id}")
            outputs.append({"id": row_id, "truth": int(y), "choice": int(prediction),
                            "lookup_logp": float(log_l), "update_logp": float(log_u),
                            "p_update": float(p), "candidate_mass": float(m),
                            "text": generated["text"], "text_outcome": outcome})
        metrics = binary_metrics(truth, candidate)
        if not np.isclose(metrics["accuracy"], data["candidate_label_accuracy"]):
            raise ValueError(f"candidate choice does not match the recorded score for {stage}")
        counts, means, rates, brier = reliability_bins(truth, p_update, EDGES)
        thresholds = {}
        for t in (0.2, 0.5, 0.8):
            result = binary_metrics(truth, p_update > t)
            thresholds[str(t)] = {**result, "cost": result["fp"] + 4 * result["fn"]}
        stages[stage] = {
            "choice_metrics": metrics, "rows": outputs,
            "text_counts": {name: sum(row["text_outcome"] == name for row in outputs)
                            for name in ("correct", "wrong label", "invalid")},
            "bins": [{"count": int(count), "mean": float(mean), "observed": float(rate)}
                     for count, mean, rate in zip(counts, means, rates) if count],
            "brier": brier, "thresholds": thresholds,
        }
    a = np.array([row["choice"] == row["truth"] for row in stages["base"]["rows"]])
    b = np.array([row["choice"] == row["truth"] for row in stages["sft"]["rows"]])
    differences = b.astype(int) - a.astype(int)
    family_ids = sorted(set(families))
    grouped = np.stack([differences[families == family] for family in family_ids])
    if not np.all(grouped == grouped[:, :1]):
        raise ValueError("the exact family calculation here needs identical differences within each family")
    return {"stages": stages, "size": len(truth), "family_ids": family_ids,
            "family_differences": grouped.tolist(), "point": float(differences.mean()),
            "both_correct": int((a & b).sum()), "base_only": int((a & ~b).sum()),
            "sft_only": int((~a & b).sum()), "both_wrong": int((~a & ~b).sum()),
            "row_resampling": resampling_distribution(differences),
            "family_resampling": resampling_distribution(grouped[:, 0])}


def context_evaluation(record):
    observations = record["observations"]
    groups = {}
    for count in RECORD_COUNTS:
        for placement in (*POSITIONS, "fixed suffix"):
            rows = [row for row in observations if row["records"] == count and row["placement"] == placement]
            pairs = {(row["report"], row["gold"]): row for row in rows}
            if len(rows) != 2 * REPORTS or set(pairs) != {(i, y) for i in range(1, REPORTS + 1) for y in (0, 1)}:
                raise ValueError(f"incomplete paired context cell: {count}, {placement}")
            for field in ("input_tokens", "context_tokens", "distance"):
                if len({row[field] for row in rows}) != 1:
                    raise ValueError(f"unexpected difference in {field} within a cell")
            groups[count, placement] = {
                "correct": sum(row["gold"] == row["choice"] for row in rows),
                "greedy_correct": sum(row["greedy_correct"] for row in rows),
                "input_tokens": rows[0]["input_tokens"], "context_tokens": rows[0]["context_tokens"],
                "distance": rows[0]["distance"],
                "first_report": [pairs[1, y]["p_blue"] for y in (0, 1)],
                "probability_changes": [pairs[i, 1]["p_blue"] - pairs[i, 0]["p_blue"]
                                        for i in range(1, REPORTS + 1)],
            }
    if len({groups[n, "fixed suffix"]["distance"] for n in RECORD_COUNTS}) != 1:
        raise ValueError("fixed suffix experiment changed distance")
    return groups


def draw(adaptation, context, raw_context, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.ticker import PercentFormatter

    blue, green, orange, gray = "#296799", "#238165", "#bd6026", "#c5cad1"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#233142"})

    def save(fig, name):
        fig.savefig(destination / f"ch17_pretrained_{name}.png", dpi=190, facecolor="white")
        plt.close(fig)

    stages = adaptation["stages"]
    n = adaptation["size"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.75), sharey=True, layout="constrained")
    choice = [round(n * stages[name]["choice_metrics"]["accuracy"]) for name in STAGES]
    texts = [stages[name]["text_counts"] for name in STAGES]
    for ax, title, counts in (
            (axes[0], "(a) Choose between two complete labels",
             [choice, [n - x for x in choice], [0] * len(STAGES)]),
            (axes[1], "(b) Generate a label as text",
             [[row[k] for row in texts] for k in ("correct", "wrong label", "invalid")])):
        bottom = np.zeros(len(STAGES))
        for heights, color, label in zip(counts, (green, orange, gray), ("Correct", "Wrong label", "Invalid text")):
            ax.bar(NAMES, heights, width=0.63, bottom=bottom, color=color, label=label, zorder=3)
            for i, (low, height) in enumerate(zip(bottom, heights)):
                if height:
                    ax.text(i, low + height / 2, str(height), ha="center", va="center",
                            color="#203040" if color == gray else "white", fontweight="bold")
            bottom += heights
        ax.set(title=title, ylim=(0, n + 1), yticks=np.arange(0, n + 1, 8))
        ax.grid(axis="y", alpha=0.24, zorder=0)
    axes[0].set_ylabel("Number of held-out messages")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    save(fig, "answer_rules")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
    axes[0].plot([0, 1], [0, 1], ls=":", color="#788394", label="Equal prediction and rate")
    for stage, color, marker, label in (("base", blue, "o", "Published base"), ("sft", green, "s", "After SFT")):
        points = stages[stage]["bins"]
        axes[0].scatter([p["mean"] for p in points], [p["observed"] for p in points],
                        color=color, marker=marker, s=48, label=label, zorder=3)
        for point in points:
            x, y = point["mean"], point["observed"]
            offset = (-7, 5) if x > 0.8 else (5, -11 if .4 < x < .6 else 5)
            axes[0].annotate(f"n={point['count']}", (x, y),
                             xytext=offset, textcoords="offset points",
                             ha="right" if x > 0.8 else "left", fontsize=9, color=color)
    axes[0].set(title="(a) Reliability of UPDATE scores", xlabel="Mean candidate probability of UPDATE",
                ylabel="Observed UPDATE fraction", xlim=(-0.035, 1.035), ylim=(-0.04, 1.04),
                xticks=[0, .2, .4, .6, .8, 1], yticks=[0, .2, .4, .6, .8, 1])
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)

    rows = stages["base"]["rows"]
    p = np.array([row["p_update"] for row in rows])
    y = np.array([row["truth"] for row in rows], dtype=bool)
    thresholds = np.unique(np.r_[0, p, 1])
    predicted = p[None, :] > thresholds[:, None]
    axes[1].step(thresholds, (predicted & ~y).sum(axis=1), where="post", color=orange, label="False alarms")
    axes[1].step(thresholds, (~predicted & y).sum(axis=1), where="post", color=blue, label="Missed updates")
    for t in (.2, .5):
        axes[1].axvline(t, color="#8792a0", ls=":", lw=1)
    axes[1].set(title="(b) Threshold sweep on the base model", xlabel="Threshold for calling UPDATE",
                ylabel="Number of errors", xlim=(0, 1), ylim=(0, 17),
                xticks=[0, .2, .5, .8, 1], yticks=[0, 4, 8, 12, 16])
    axes[1].legend(loc="center right", fontsize=9, frameon=False)
    save(fig, "calibration")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.25), layout="constrained", width_ratios=[1, 1.48])
    values = np.asarray(adaptation["family_differences"])
    cmap = ListedColormap([orange, "#e9edf2", green])
    axes[0].imshow(values, cmap=cmap, norm=BoundaryNorm([-1.5, -.5, .5, 1.5], cmap.N), aspect="auto")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            axes[0].text(j, i, {1: "+1", 0: "0", -1: "−1"}[value], ha="center", va="center",
                         color="white" if value else "#314154", fontweight="bold")
    axes[0].set(title="(a) SFT minus base, by message", xlabel="Account ID", ylabel="Wording family",
                xticks=np.arange(4), xticklabels=[100, 101, 102, 103], yticks=np.arange(8),
                yticklabels=[f"Lookup {i}" for i in range(1, 5)] + [f"Update {i}" for i in range(1, 5)])
    row = adaptation["row_resampling"]
    family = adaptation["family_resampling"]
    axes[1].bar(row["support"], row["mass"], width=0.024, color=blue, alpha=0.68, label="32 message draws")
    axes[1].vlines(family["support"], 0, family["mass"], color=orange, linewidth=1)
    axes[1].scatter(family["support"], family["mass"], color=orange, s=21, label="8 family draws", zorder=3)
    axes[1].axvline(adaptation["point"], color="#394552", ls=":", lw=1)
    axes[1].set(title="(b) Exact resampling distributions", xlabel="Resampled accuracy gap (percentage points)",
                ylabel="Probability of resampled gap", xlim=(-1.05, 1.05), ylim=(0, None),
                xticks=[-1, -.5, 0, .25, .5, 1], xticklabels=["−100", "−50", "0", "+25", "+50", "+100"])
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_yticks([0, .05, .10, .15, .20])
    axes[1].legend(loc="upper left", fontsize=9, frameon=False)
    save(fig, "paired_results")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), layout="constrained", width_ratios=[1, 1.16])
    grid = np.array([[context[n, p]["correct"] for p in POSITIONS] for n in RECORD_COUNTS])
    axes[0].imshow(grid, cmap="Blues", vmin=0, vmax=2 * REPORTS, aspect="auto")
    for i, row in enumerate(grid):
        for j, hits in enumerate(row):
            axes[0].text(j, i, f"{hits}/{2 * REPORTS}", ha="center", va="center",
                         color="white" if hits > 9 else "#193453", fontweight="bold", fontsize=13)
    lengths = [context[n, "fixed suffix"]["input_tokens"] for n in RECORD_COUNTS]
    axes[0].set(title="(a) Move the account record", xlabel="Placement of the evidence",
                ylabel="Total input tokens", xticks=np.arange(3), xticklabels=[p.title() for p in POSITIONS],
                yticks=np.arange(3), yticklabels=lengths)
    hits = [context[n, "fixed suffix"]["correct"] for n in RECORD_COUNTS]
    axes[1].axhline(REPORTS, color="#758293", ls=":", label="Constant choice on balanced cases")
    axes[1].plot(lengths, hits, "o-", color=green, lw=2, ms=6)
    for x, count in zip(lengths, hits):
        axes[1].annotate(f"{count}/{2 * REPORTS}", (x, count), xytext=(0, 7 if count >= REPORTS else -13),
                         textcoords="offset points", ha="center", color=green)
    distance = context[RECORD_COUNTS[0], "fixed suffix"]["distance"]
    axes[1].set(title=f"(b) Keep the {distance}-token suffix fixed", xlabel="Total input tokens",
                ylabel=f"Correct choices out of {2 * REPORTS}", ylim=(0, 2 * REPORTS + .5),
                xticks=lengths, yticks=[0, 4, 8, 12, 16])
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, "context_grid")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.85), layout="constrained")
    positions = np.arange(3)
    colors = np.array([context[64, p]["first_report"] for p in POSITIONS])
    axes[0].axhline(.5, color="#758293", ls=":", lw=1)
    for i, pair in enumerate(colors):
        axes[0].plot([i, i], pair, color="#abb5c1", lw=1.5)
    axes[0].scatter(positions, colors[:, 0], color=orange, marker="o", label="Record says red", zorder=3)
    axes[0].scatter(positions, colors[:, 1], color=blue, marker="s", label="Record says blue", zorder=3)
    absent = next(row for row in raw_context["observations"] if row["report"] == 1 and row["placement"] == "absent")
    axes[0].scatter([3], [absent["p_blue"]], color="#4c5561", marker="x", label="Key absent", zorder=3)
    axes[0].set(title="(a) Same report, two possible codes", ylabel="Probability of blue within the pair",
                xticks=np.arange(4), xticklabels=["Early", "Middle", "Late", "Absent"], ylim=(.29, .69))
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    shifts = np.array([context[64, p]["probability_changes"] for p in POSITIONS])
    axes[1].axhline(0, color="#758293", ls=":", lw=1)
    for i, group in enumerate(shifts):
        axes[1].scatter(i + np.linspace(-.17, .17, REPORTS), group, color=blue, alpha=.75, s=24)
    axes[1].scatter(positions, shifts.mean(axis=1), color=green, marker="D", s=43, label="Mean of eight reports", zorder=4)
    axes[1].set(title="(b) Effect of changing red to blue", ylabel="Change in blue probability",
                xticks=positions, xticklabels=[p.title() for p in POSITIONS])
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)
    save(fig, "evidence_flip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptation-run", type=Path, default=ADAPTATION_RUN)
    parser.add_argument("--context-run", type=Path, default=CONTEXT_RUN)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch17-pretrained"))
    args = parser.parse_args()
    adaptation_bytes = args.adaptation_run.read_bytes()
    context_bytes = args.context_run.read_bytes()
    source, raw_context = json.loads(adaptation_bytes), json.loads(context_bytes)
    for key in ("model", "revision", "weights_sha256"):
        if source[key] != raw_context[key]:
            raise ValueError(f"the base checkpoint differs in {key}")
    adaptation = adaptation_evaluation(source)
    context = context_evaluation(raw_context)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(adaptation, context, raw_context, args.output_dir)
    result = {"model": source["model"], "revision": source["revision"],
              "adaptation_input_sha256": hashlib.sha256(adaptation_bytes).hexdigest(),
              "context_input_sha256": hashlib.sha256(context_bytes).hexdigest(),
              "adaptation": adaptation,
              "context": [{"records": n, "placement": p, **values} for (n, p), values in context.items()]}
    (args.output_dir / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved five figures and per-case calculations to {args.output_dir}")


if __name__ == "__main__":
    main()
