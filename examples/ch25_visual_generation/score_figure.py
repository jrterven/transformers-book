"""Draw the analytic one-dimensional score example used in Appendix F."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def gaussian_density_and_score(values, mean=2.0, sigma=1.0):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    values = np.asarray(values, dtype=np.float64)
    density = np.exp(-0.5 * ((values - mean) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    return density, -(values - mean) / sigma ** 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-1, 5, 501)
    density, score = gaussian_density_and_score(x)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.65), layout="constrained")
    axes[0].plot(x, density, color="#3976AF", linewidth=2.5)
    axes[0].fill_between(x, density, color="#3976AF", alpha=.08)
    axes[0].axvline(2, color="#687580", linestyle="--", linewidth=1)
    for point in (0, 1, 3, 4):
        direction = np.sign(2 - point)
        axes[0].annotate("", xy=(point + .4 * direction, .025), xytext=(point, .025),
                         arrowprops={"arrowstyle": "->", "color": "#46856E", "lw": 2})
    axes[0].set(xlabel="Coordinate u", ylabel="Density p(u)", ylim=(0, .45),
                title="A Gaussian density: mean 2, variance 1")
    axes[1].plot(x, score, color="#46856E", linewidth=2.5)
    axes[1].axhline(0, color="#687580", linewidth=1)
    axes[1].axvline(2, color="#687580", linestyle="--", linewidth=1)
    axes[1].scatter([0, 3], [2, -1], color="#3976AF", s=45, zorder=3)
    axes[1].annotate("s(0) = 2", (0, 2), xytext=(.3, 2.3), fontsize=12)
    axes[1].annotate("s(3) = -1", (3, -1), xytext=(3.2, -.5), fontsize=12)
    axes[1].set(xlabel="Coordinate u", ylabel="Score s(u)", ylim=(-3.5, 3.5),
                title="Derivative of log density: s(u) = 2 - u")
    for ax in axes:
        ax.set_xlim(-1, 5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    path = args.output / "app_f_score_intro.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    (args.output / "score-manifest.json").write_text(json.dumps({
        "mean": 2, "variance": 1, "formula": "score(u) = 2-u",
        "figure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
