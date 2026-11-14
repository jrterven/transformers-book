"""Render the measured image and video figures from saved observations."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .demo_media import asset, image_pair, sampled_frames


EXAMPLE = Path(__file__).resolve().parent
INK = "#182f42"
BLUE = "#2c709a"
TEAL = "#168278"
GOLD = "#eab952"
GRAY = "#536c7d"
GRID = "#dbe6ec"
COLORS = LinearSegmentedColormap.from_list("cosine", ["#f4f4f1", "#accbd9", "#468f9f", "#214b69"])


def save(fig, output, name):
    path = output / f"ch22_pretrained_{name}.png"
    fig.savefig(path, dpi=185, facecolor="white", bbox_inches="tight", pad_inches=.07)
    plt.close(fig)


def box(ax, rc, size=1, color=GOLD, style="-", width=2.0):
    r, c = rc
    ax.add_patch(Rectangle((c * size - .5, r * size - .5), size, size,
                           fill=False, edgecolor=color, linewidth=width, linestyle=style))


def photo(ax, image, title, rc, patch, other=None):
    ax.imshow(image)
    ax.set_title(title, loc="left", fontweight="bold", pad=7)
    ax.set_xticks(np.arange(patch, image.shape[1], patch) - .5, minor=True)
    ax.set_yticks(np.arange(patch, image.shape[0], patch) - .5, minor=True)
    ax.grid(which="minor", color="white", alpha=.30, linewidth=.6)
    ax.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)
    box(ax, rc, patch)
    if other:
        box(ax, other, patch, color="white", style="--", width=1.2)


def image_figures(record, original, shifted, output):
    rc = tuple(record["query_rc"])
    target = tuple(record["target_rc"])
    patch = record["patch_size"]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.78), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1.30]})
    photo(axes[0], original, "Original", rc, patch)
    photo(axes[1], shifted, "32 pixels right", target, patch, rc)
    ax = axes[2]
    values = record["projection_channel_sums"] + [record["projection_bias"]]
    positions = np.arange(len(values))
    ax.barh(positions, values, color=["#b85552", TEAL, BLUE, "#8294a2"], height=.55)
    ax.axvline(0, color=GRAY, linewidth=.9)
    ax.set_title("Projection coordinate 0", loc="left", fontweight="bold", pad=7)
    ax.set_yticks(positions, ["Red", "Green", "Blue", "Bias"])
    ax.invert_yaxis()
    ax.set_xlim(-.37, .57)
    ax.set_xticks([-.2, 0, .2, .4])
    for row, value in enumerate(values):
        offset = .022 if value >= 0 else -.022
        ax.text(value + offset, row, f"{value:+.3f}", ha="left" if value >= 0 else "right", va="center")
    ax.set_xlabel(f"Sum = {record['projection_coordinate']:.3f}", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, output, "patch_projection")

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.15), layout="constrained")
    photo(axes[0], shifted, "Shifted photo", target, patch, rc)
    arrays = [record["original_cosine"], record["shifted_cosine"]]
    lim = min(0.0, min(np.asarray(a).min() for a in arrays))
    for ax, values, title, expected, ref in zip(
        axes[1:], arrays, ["Search in original", "Search in shift"], [rc, target], [None, rc], strict=True,
    ):
        mesh = ax.imshow(values, cmap=COLORS, vmin=lim, vmax=1, interpolation="nearest")
        ax.set_title(title, loc="left", fontweight="bold", pad=7)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_yticks([0, 5, 10, 15])
        ax.set_xlabel("Patch column")
        box(ax, expected, color=GOLD)
        if ref:
            box(ax, ref, color="white", style="--", width=1.2)
    cbar = fig.colorbar(mesh, ax=axes[1:], fraction=.035, pad=.025)
    cbar.set_label("Cosine")
    save(fig, output, "patch_similarity")


def filmstrip(clip, frames, crop, output):
    fig, axes = plt.subplots(2, 4, figsize=(10, 3.45))
    selected = [0, 5, 10, 15]
    for row, (indices, label) in enumerate(((selected, "Original"), (selected[::-1], "Reversed"))):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(frames[idx])
            x, y, w, h = crop
            ax.add_patch(Rectangle((x, y), w, h, fill=False, linestyle="--",
                                   linewidth=.8, edgecolor="#536c7d"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{clip['timestamps_s'][idx]:.2f} s", pad=4, color=GRAY)
            if col == 0:
                ax.set_ylabel(label, rotation=0, labelpad=10, ha="right", va="center", fontweight="bold")
    fig.subplots_adjust(left=.12, right=.99, top=.93, bottom=.05, hspace=.16, wspace=.035)
    save(fig, output, "fan_frames")


def video_scores(video, output):
    fig, (ax, audit) = plt.subplots(1, 2, figsize=(10.1, 3.64), gridspec_kw={"width_ratios": [1.24, 1]})
    ax.axvspan(-7, 0, color="#eff5f9", zorder=0)
    ax.axvspan(0, 7, color="#edf7f3", zorder=0)
    ax.axvline(0, color=GRAY, linewidth=1, zorder=1)
    ax.set_xlim(-7, 7)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_yticks([5, 3, 1], ["First unfold", "Second unfold", "Fold"])
    ax.set_ylim(.2, 6.15)
    ax.set_xlabel("Logit for unfolding minus logit for folding")
    ax.text(-6.75, 5.91, "Favors folding", color=BLUE, fontsize=9, va="center")
    ax.text(6.75, 5.91, "Favors unfolding", color=TEAL, fontsize=9, ha="right", va="center")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for y, clip in zip([5, 3, 1], video["clips"], strict=True):
        original = clip["original"]["temporal"]["unfolding_minus_folding"]
        reverse = clip["reversed"]["temporal"]["unfolding_minus_folding"]
        still = clip["original"]["frame_average"]["unfolding_minus_folding"]
        ax.plot([original, reverse], [y, y], color="#94aebb", linewidth=1.4, zorder=2)
        ax.scatter(original, y, color=TEAL, marker="o", s=73, zorder=3)
        ax.scatter(reverse, y, color=BLUE, marker="s", s=69, zorder=3)
        ax.scatter(still, y - .38, color="#686e75", marker="D", s=30, zorder=3)
        for value in (original, reverse):
            ax.text(value, y + .28, f"{value:+.2f}", ha="center", color=INK, fontsize=8.5)
    ax.scatter([], [], color=TEAL, marker="o", s=55, label="Original")
    ax.scatter([], [], color=BLUE, marker="s", s=55, label="Reversed")
    ax.scatter([], [], color="#686e75", marker="D", s=33, label="Frame mean (both)")
    ax.legend(loc="lower left", bbox_to_anchor=(-.09, -.31), ncol=3, frameon=False,
              handletextpad=.2, columnspacing=.8, fontsize=9)

    audit.axis("off")
    audit.text(0, .98, "What counts as correct?", va="top", weight="bold", fontsize=12)
    headers = ["Scoring rule", "Full clip", "Frame mean"]
    full, still = (video["pair_metrics"][key] for key in ("temporal", "frame_average"))
    rows = [("Two labels\nPer view", "correct", "views"),
            ("Two labels\nPer pair", "both_correct", "pairs"),
            ("174 labels\nTop-1", "full_vocabulary_top1_correct", "views")]
    cells = [[name, f"{full[key]}/{full[total]}", f"{still[key]}/{still[total]}"] for name, key, total in rows]
    table = audit.table(cellText=cells, colLabels=headers, cellLoc="left", colLoc="left",
                        colWidths=[.48, .23, .29], bbox=[0, .36, 1, .46])
    table.auto_set_font_size(False)
    table.set_fontsize(9.3)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#e1ecf3" if i == 0 else ("#f4f7f9" if i % 2 else "#ecf2f6"))
        cell.get_text().set_color(INK)
        if i == 0 or j == 1:
            cell.get_text().set_weight("bold")
    fig.subplots_adjust(left=.15, right=.99, top=.98, bottom=.22, wspace=.25)
    save(fig, output, "video_direction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=EXAMPLE / "PRETRAINED_RUN.json")
    parser.add_argument("--media-dir", type=Path, default=Path("output/ch22-media"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch22-pretrained-figures"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
                               "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.titlecolor": INK,
                               "xtick.color": GRAY, "ytick.color": GRAY})
    run = json.loads(args.record.read_text(encoding="utf-8"))
    image, video = run["image"], run["video"]
    prep = image["preprocess"]
    original, shifted = image_pair(asset(args.media_dir, "astronaut.png", offline=args.offline),
                                   prep["size"], prep["shift_pixels"], prep["fill_rgb"])
    if [hashlib.sha256(frame.tobytes()).hexdigest() for frame in (original, shifted)] != image["rgb_sha256"]:
        raise ValueError("Figure image pixels differ from the inference record")
    clip = video["clips"][1]
    frames, times = sampled_frames(asset(args.media_dir, "fan.webm", offline=args.offline), clip["indices"])
    hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
    if times != clip["timestamps_s"] or hashes != clip["rgb_frame_sha256"]:
        raise ValueError("Figure video frames differ from the inference record")
    image_figures(image, original, shifted, args.output_dir)
    height, width = frames.shape[1:3]
    short_edge = video["preprocess"]["resize_short_edge"]
    resized_width = int(width * short_edge / height)
    crop_height, crop_width = video["preprocess"]["crop"]
    shown_crop = ((resized_width - crop_width) // 2 * width / resized_width - .5,
                  (short_edge - crop_height) // 2 * height / short_edge - .5,
                  crop_width * width / resized_width, crop_height * height / short_edge)
    filmstrip(clip, frames, shown_crop, args.output_dir)
    video_scores(video, args.output_dir)
    print(f"Rendered four measured figures in {args.output_dir}")


if __name__ == "__main__":
    main()
