"""Render Chapter 18's figures from the recorded encoder experiments."""

import argparse
import json
from pathlib import Path

import numpy as np


RECORD = Path(__file__).with_name("PRETRAINED_RUN.json")


def draw(record, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import PercentFormatter

    blue, green, orange = "#286999", "#247b63", "#ba6028"
    ink, gray = "#263546", "#8794a2"
    plt.rcParams.update({"font.size": 10.5, "axes.labelcolor": ink,
                         "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(destination / f"ch18_pretrained_{name}.png", dpi=190, facecolor="white")
        plt.close(fig)

    cos = np.array(record["context"]["cosines"])
    layer = np.arange(len(cos))
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.65), layout="constrained", width_ratios=[1.45, 1])
    axes[0].plot(layer, cos[:, 0, 1], "o-", ms=3.5, color=blue, label="River pair")
    axes[0].plot(layer, cos[:, 2, 3], "s-", ms=3.5, color=green, label="Finance pair")
    cross = cos[:, :2, 2:].mean(axis=(1, 2))
    axes[0].plot(layer, cross, "^-", ms=3.5, color=orange, label="Across contexts (mean)")
    axes[0].set(title="(a) Same token, through BERT", xlabel="Encoder block (0 is embedding output)",
                ylabel="Cosine similarity", xticks=[0, 2, 4, 6, 8, 10, 12], ylim=(min(.5, cross.min() - .04), 1.02))
    axes[0].grid(axis="y", alpha=.2)
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=False)
    final = cos[-1]
    axes[1].imshow(final, cmap="Blues", vmin=0, vmax=1)
    for i, row in enumerate(final):
        for j, val in enumerate(row):
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if val > .7 else ink, fontsize=11, fontweight="bold")
    labels = ["River 1", "River 2", "Finance 1", "Finance 2"]
    axes[1].set(title="(b) Cosines at block 12", yticks=range(4), yticklabels=labels)
    axes[1].set_xticks(range(4), labels, rotation=29, ha="right")
    save(fig, "context_geometry")

    intent = record["intent"]
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.5), layout="constrained", width_ratios=[1.14, 1])
    p = np.array([intent["probes"][pool]["held_update_p"] for pool in ("cls", "mean")])
    gold = np.array([row["label"] for row in intent["held"]])
    heat = axes[0].imshow(p, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0].axvline(15.5, color="white", ls=":", lw=1.2)
    for i, j in zip(*np.nonzero((p > .5) != gold[None, :])):
        axes[0].add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="none", edgecolor="#ffb159", lw=1.6))
    axes[0].set(title="(a) Two heads on frozen BERT features", xlabel="Held-out message index",
                yticks=[0, 1], yticklabels=["[CLS]", "Content mean"], xticks=[0, 7, 15, 23, 31],
                xticklabels=[1, 8, 16, 24, 32])
    fig.colorbar(heat, ax=axes[0], fraction=.045, ticks=[0, .5, 1], label="UPDATE probability")
    paths = intent["padding"]["paths"]
    margins = [paths[key]["update_margin"] for key in ("short_correct", "padded_correct", "pool_pad_rows", "attend_pad_keys")]
    axes[1].barh(range(4), margins, color=[green, green, orange, orange], height=.54)
    axes[1].set(title="(b) Padding and the update margin", xlabel="UPDATE logit minus LOOKUP logit",
                yticks=range(4), yticklabels=["Short input", "Correct padding", "Pool PAD rows", "Attend to PAD"],
                xlim=(0, max(margins) * 1.21))
    axes[1].invert_yaxis()
    for i, v in enumerate(margins):
        axes[1].text(v + .4, i, f"{v:.2f}", va="center", fontsize=9)
    save(fig, "sentence_pooling")

    nli = record["nli"]
    fig, ax = plt.subplots(figsize=(9.7, 3.65), layout="constrained")
    probs = np.array([row["probabilities"] for row in nli["rows"]])
    y = np.arange(len(probs))
    left = np.zeros(len(y))
    for i, (label, color) in enumerate(zip(nli["labels"], (orange, green, blue))):
        values = probs[:, i]
        ax.barh(y, values, left=left, color=color, height=.64, label=label.title())
        for j, v in enumerate(values):
            if v > .15:
                ax.text(left[j] + v / 2, j, f"{v:.1%}", color="white", ha="center", va="center", fontweight="bold")
        left += values
    row_names = [row["case"] + (" *" if row["prediction"] != row["gold"] else "") for row in nli["rows"]]
    ax.set(title="Native sentence-pair probabilities", xlabel="Probability across the three labels",
           yticks=y, yticklabels=row_names, xlim=(0, 1))
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    fig.legend(loc="outside lower center", ncols=3, frameon=False)
    save(fig, "nli_pairs")

    ner = record["ner"]
    original = next(row for row in ner["rows"] if row["case"] == 2 and row["casing"] == "original")
    lower = next(row for row in ner["rows"] if row["case"] == 2 and row["casing"] == "lower")
    labels = ner["labels"]
    logits = np.array(original["logits"])
    scores = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = scores / scores.sum(axis=1, keepdims=True)
    pieces = original["pieces"]
    fig, axes = plt.subplots(2, 1, figsize=(10.35, 4.25), sharex=True, layout="constrained", height_ratios=[5.6, 1])
    image = axes[0].imshow(probs.T, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set(yticks=range(len(labels)), yticklabels=labels, ylabel="Native tag")
    top = axes[0].secondary_xaxis("top")
    top.set_xticks(range(len(pieces)), [str(i) if i >= 0 else "—" for i in original["word_ids"]])
    top.set_xlabel("Tokenizer word ID for each piece", labelpad=7)
    for x in original["first_positions"]:
        axes[0].add_patch(Rectangle((x - .5, -.5), 1, len(labels), fill=False, edgecolor=green, lw=1.25))
    fig.colorbar(image, ax=axes, fraction=.026, ticks=[0, .5, 1], label="Native tag probability")
    axes[1].set(ylim=(0, 1), yticks=[], ylabel="Target")
    axes[1].spines["left"].set_visible(False)
    for x, target in enumerate(original["targets"]):
        axes[1].add_patch(Rectangle((x - .46, .15), .92, .70, color="#e6f3ed" if target >= 0 else "#edf0f3"))
        axes[1].text(x, .50, labels[target] if target >= 0 else "−100", ha="center", va="center", fontsize=8.5, color=ink)
    axes[1].set_xticks(range(len(pieces)), pieces, rotation=25, ha="right")
    axes[1].set_xlabel("Input piece")
    save(fig, "ner_alignment")

    fig, ax = plt.subplots(figsize=(10.3, 2.8), layout="constrained")
    words = original["words"]
    views = [original["gold_spans"], original["first_spans"], lower["first_spans"]]
    for i in range(len(words) + 1):
        ax.axvline(i, color="#dde3e9", lw=.8)
    for i, spans in enumerate(views):
        for start, end, kind in spans:
            color = (blue, green, orange)[i]
            ax.add_patch(Rectangle((start, i - .27), end - start, .54, color=color))
            ax.text((start + end) / 2, i, f"{kind} [{start}, {end})", color="white",
                    ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set(title="Exact spans on the same seven word positions", xlim=(0, len(words)),
           ylim=(2.65, -.65), yticks=[0, 1, 2], yticklabels=["Gold spans", "Original casing", "Lowercase input"],
           xticks=range(len(words) + 1), xlabel="Word boundary index")
    words_axis = ax.secondary_xaxis("top")
    words_axis.set_xticks(np.arange(len(words)) + .5, words)
    words_axis.tick_params(axis="x", length=0, pad=5, labelsize=9)
    save(fig, "span_boundaries")

    summaries = [ner["summary"][name]["first_piece"] for name in ("original", "lower")]
    fig, axes = plt.subplots(1, 2, figsize=(9.95, 3.5), layout="constrained", width_ratios=[1.5, 1])
    for i, (summary, label, color) in enumerate(zip(summaries, ("Original casing", "Lowercase input"), (blue, orange))):
        values = [summary[key] for key in ("word_accuracy", "precision", "recall", "f1")]
        m, p, g = (summary[key] for key in ("matches", "predicted", "gold"))
        counts = [(summary["word_matches"], summary["words"]), (m, p), (m, g), (2 * m, p + g)]
        left = -.185 if i == 0 else .185
        bars = axes[0].bar(np.arange(4) + left, values, width=.34, color=color, label=label)
        axes[0].bar_label(bars, labels=[f"{a}/{b}" for a, b in counts], padding=3, fontsize=8.2)
        types = [summary["by_type"][k] for k in ("PER", "LOC", "ORG")]
        ratios = [x["matches"] / x["gold"] for x in types]
        bars = axes[1].bar(np.arange(3) + left, ratios, width=.34, color=color)
        axes[1].bar_label(bars, labels=[f"{x['matches']}/{x['gold']}" for x in types], padding=3, fontsize=8.2)
    axes[0].set(title="(a) Two ways to score the same tags", xticks=range(4),
                xticklabels=["Word accuracy", "Span precision", "Span recall", "Span F1"], ylabel="Score")
    axes[1].set(title="(b) Exact recall by type", xticks=range(3), xticklabels=["PER", "LOC", "ORG"])
    for ax in axes:
        ax.set(ylim=(0, 1.16), yticks=[0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncols=2, frameon=False)
    save(fig, "casing_scores")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch18-real"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(json.loads(args.record.read_text(encoding="utf-8")), args.output_dir)
    print(f"Saved six measured figures to {args.output_dir}")


if __name__ == "__main__":
    main()
