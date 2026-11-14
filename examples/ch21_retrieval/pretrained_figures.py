"""Plot the recorded Chapter 21 search and grounded-generation experiments."""

import argparse
import json
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np


RECORD = Path(__file__).with_name("PRETRAINED_RUN.json")
INK = "#26374a"
BLUE = "#236698"
GREEN = "#21785f"
ORANGE = "#b85b27"
GRAY = "#8795a4"
GRID = "#dce5ec"
FOCUS = "renew_paraphrase"
FOCUS_LABELS = {
    "P01": "Renewal location",
    "P02": "Authorization dates",
    "P03": "Sticker replacement",
    "P04": "Visitor pass",
    "P11": "Printer service",
}
QUESTION_LABELS = {
    "renew_literal": "Renewal, literal",
    FOCUS: "Renewal, paraphrased",
    "replace_sticker": "Replace sticker",
    "visitor_pass": "Visitor pass",
    "scanner_two_facts": "Scanner and key",
    "return_book": "Return a book",
    "toner_two_facts": "Toner location and hours",
    "pickup_parcel": "Collect a parcel",
}


def save(fig, destination, name):
    fig.savefig(destination / f"ch21_pretrained_{name}.png", dpi=210, facecolor="white")
    plt.close(fig)


def search_scores(row, destination):
    ids = row["dense"][:5]
    fig, (lexical, dense) = plt.subplots(
        1, 2, figsize=(8.4, 3.0), layout="constrained", sharey=True)
    for ax, key, title, axis_label, limit in (
        (lexical, "bm25_scores", "(a) Lexical search", "BM25 score", 11.7),
        (dense, "cosine_scores", "(b) Qwen3 embeddings", "Cosine similarity", .96),
    ):
        for i, pid in enumerate(ids):
            score = row[key][pid]
            color = GREEN if pid == "P01" else BLUE
            ax.barh(i, score, color=color, height=.54)
            ax.text(score + .025 * limit, i, f"{score:.3f}", va="center", color=INK, fontsize=10)
        ax.axhline(2.5, color=GRAY, ls=":", lw=1.2)
        ax.set(title=title, xlabel=axis_label, xlim=(0, limit))
    lexical.set_yticks(range(len(ids)), [f"{pid}  {FOCUS_LABELS[pid]}" for pid in ids])
    lexical.invert_yaxis()
    dense.tick_params(axis="y", left=False)
    dense.spines["left"].set_visible(False)
    save(fig, destination, "search_scores")


def rerank_paths(row, destination):
    before, after = row["fused"], row["reranked"]
    fig, ax = plt.subplots(figsize=(8.4, 2.9), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(-.55, len(before) - .25))
    ax.invert_yaxis()
    ax.axis("off")
    ax.text(.015, -.45, "Fusion shortlist", fontweight="bold", va="center", fontsize=12)
    ax.text(.635, -.45, "Qwen3 pair scores", fontweight="bold", va="center", fontsize=12)
    for left, pid in enumerate(before):
        right = after.index(pid)
        color = GREEN if pid == "P01" else GRAY
        ax.plot([.43, .60], [left + .22, right + .22], color=color, lw=2.2 if pid == "P01" else 1.15)
        ax.scatter([.43, .60], [left + .22, right + .22], color=color, s=23, zorder=3)
        weight = "bold" if pid == "P01" else "normal"
        ax.text(.015, left + .22, f"{left + 1}   {pid}   {FOCUS_LABELS[pid]}",
                va="center", fontweight=weight)
        ax.text(.635, right + .22, f"{right + 1}   {pid}     {row['pair_scores'][pid]['p_yes']:.4f}",
                va="center", fontweight=weight)
    save(fig, destination, "rerank_paths")


def retrieval_recall(record, destination):
    questions = [q for q in record["fixture"]["questions"] if q["evidence"]]
    stages = ("bm25", "dense", "fused", "reranked", "packed")
    matrix = []
    for q in questions:
        key = q["id"]
        gold = set(q["evidence"])
        row = record["retrieval"][key]
        ids = [row[s][:3] for s in stages[:-1]] + [record["answers"][key]["packed"]]
        matrix.append([len(gold.intersection(ranking)) / len(gold) for ranking in ids])
    fig, ax = plt.subplots(figsize=(8.4, 3.45), layout="constrained")
    cmap = ListedColormap(["#f7e3d5", "#f4eedb", "#d4eee3"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, "½" if value == .5 else str(int(value)), ha="center", va="center", color=INK)
    names = ("BM25\nTop 3", "Dense\nTop 3", "Fusion\nTop 3", "Reranked\nTop 3", "Prompt\nAll kept")
    counts = [record["retrieval_metrics"][s]["3" if s != "packed" else "all"]["complete_questions"] for s in stages]
    ax.set_xticks(range(len(stages)), [f"{name}\n{count}/{len(questions)} complete"
                                     for name, count in zip(names, counts, strict=True)])
    ax.set_yticks(range(len(questions)), [QUESTION_LABELS[q["id"]] for q in questions])
    ax.set_xticks(np.arange(len(stages)) - .5, minor=True)
    ax.set_yticks(np.arange(len(questions)) - .5, minor=True)
    ax.grid(which="minor", color="white", lw=1.25)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, destination, "retrieval_recall")


def packing_budget(record, destination):
    runs = [record["answers"][FOCUS], record["controls"]["tight_input"]]
    palette = {"P02": "#759fc0", "P03": "#a0bbce", "P01": GREEN, "P04": "#c7d8e2", "P11": "#dce7ed"}
    fig, ax = plt.subplots(figsize=(8.4, 2.65), layout="constrained")
    for y, run in enumerate(runs):
        base = run["empty_prompt_tokens"]
        ax.barh(y, base, color="#d8dee6", height=.50)
        ax.text(base / 2, y, f"Prompt\n{base}", ha="center", va="center", fontsize=9)
        previous = base
        for step in run["packing"]:
            if not step["kept"]:
                continue
            pid, stop = step["id"], step["trial_tokens"]
            width = stop - previous
            ax.barh(y, width, left=previous, color=palette[pid], height=.50)
            ax.text(previous + width / 2, y, f"{pid}\n{width}", ha="center", va="center",
                    color="white" if pid == "P01" else INK, fontsize=9)
            previous = stop
        assert previous == run["prompt_tokens"]
        cap = run["input_cap"]
        ax.vlines(cap, y - .36, y + .36, color=INK, lw=1.3)
        ax.text(cap + 4, y, f"limit {cap}", va="center", fontsize=9)
    ax.set(xlim=(0, max(run["input_cap"] for run in runs) + 62),
           yticks=(0, 1), yticklabels=("Full budget", "Tight budget"),
           xlabel="Tokens in the formatted input")
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    save(fig, destination, "packing_budget")


def location_edit(record, destination):
    original = record["controls"]["evidence_only"]
    edited = record["controls"]["location_edited"]
    sources = [record["fixture"]["passages"]["P01"], edited["passage_edit"]["text"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.75), layout="constrained")
    for ax, name, text, run in zip(axes, ("Original notice", "Location edited"),
                                   sources, (original, edited), strict=True):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.add_patch(Rectangle((.01, .04), .98, .90, facecolor="#f7f9fb", edgecolor=GRID))
        ax.text(.055, .865, name, fontweight="bold", fontsize=12, va="top")
        ax.text(.055, .71, "Passage P01 (last sentence)", color=BLUE, fontweight="bold", fontsize=10, va="top")
        excerpt = text[text.index("For in-person renewals"):]
        ax.text(.055, .60, fill(excerpt, width=46), fontsize=10, va="top", linespacing=1.35)
        ax.text(.055, .39, "Generated answer", color=GREEN, fontweight="bold", fontsize=10, va="top")
        ax.text(.055, .30, fill(run["answer"]["text"], width=46), fontsize=10, va="top", linespacing=1.30)
    save(fig, destination, "location_edit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch21-figures"))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    data = json.loads(args.record.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row = data["retrieval"][FOCUS]
    search_scores(row, args.output_dir)
    rerank_paths(row, args.output_dir)
    packing_budget(data, args.output_dir)
    location_edit(data, args.output_dir)
    retrieval_recall(data, args.output_dir)


if __name__ == "__main__":
    main()
