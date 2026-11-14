"""Draw the measured recurrent context exchange from the saved JSON record."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(record, output):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(8.5, 3.5), layout="constrained")
        colors = ("#236594", "#b45422")
        rows = record["weights"]
        for i, (weights, name, color) in enumerate(zip(rows, ("Normal prefix", "Changed prefix"),
                                                       colors, strict=True)):
            left.bar(np.arange(len(weights)) + (i-.5)*.34, weights, .32, color=color, label=name)
        left.set_xticks(range(len(rows[0])), [f"{i+1}\n{w}" for i, w in enumerate(record["source"])])
        left.set(title="(a) Observe the two native reads", ylim=(0, 1.03),
                 ylabel="Attention weight", xlabel="Source position")
        left.legend(loc="upper left", frameon=False, fontsize=9)
        values = [v["gap"] for v in record["conditions"]]
        bars = right.bar(np.arange(4), values, color=[*colors, colors[0], colors[1]],
                         edgecolor="#183247", linewidth=.55)
        for bar in bars[2:]:
            bar.set_hatch("//")
        right.bar_label(bars, [f"{v:+.2f}" for v in values], padding=3, fontsize=10)
        right.axhline(0, color="#444444", lw=.8)
        right.set_xticks(range(4), ("Normal\nnative", "Changed\nnative",
                                   "Changed\n+ normal\ncontext", "Normal\n+ changed\ncontext"))
        right.set(title="(b) Exchange only the context", ylim=(-6.5, 8.5),
                  ylabel=r"Logit gap $z_\mathrm{green}-z_\mathrm{yellow}$")
        right.grid(axis="y", color="#d7e1e9", zorder=0)
        right.set_axisbelow(True)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(json.loads(args.record.read_text()), args.output)
