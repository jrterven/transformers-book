"""Plot the held-out length experiment from its measured JSON record."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw(record, output):
    data = record["groups"]
    lengths = [int(key) for key in data]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(8.5, 3.6), layout="constrained")
        ax.axvspan(2.83, 6.5, color="#e7f0f8", label="Training length range")
        ax.axvline(6.5, color="#8199aa", ls=":", lw=1)
        tokens = [data[str(length)]["token_accuracy"] for length in lengths]
        exact = [data[str(length)]["exact_match"] for length in lengths]
        ax.plot(lengths, tokens, color="#236594", marker="o", lw=1.9,
                label="Correct tokens with reference prefixes", zorder=3)
        ax.plot(lengths, exact, color="#b45422", marker="s", ms=5, lw=1.75, ls="--",
                label="Exact sequences under free generation")
        for length in lengths[4:]:
            group = data[str(length)]
            ax.annotate(f"{round(group['exact_match']*group['sequences'])}/{group['sequences']}",
                        (length, group["exact_match"]), xytext=(0, 8), ha="center",
                        textcoords="offset points", color="#873d1c")
        ax.set(xlim=(2.83, 9.19), ylim=(-.04, 1.10), xticks=lengths,
               xlabel="Number of source colors", ylabel="Fraction correct")
        ax.grid(axis="y", color="#d7e1e9")
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", frameon=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(json.loads(args.record.read_text()), args.output)
