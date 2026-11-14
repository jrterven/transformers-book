"""Draw Chapter 23 figures from the frozen checkpoint outputs and COCO polygons."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
import numpy as np

from .demo_data import EXAMPLE, decode_mask, load_photo, read_fixture, reference_instances


INK = "#193143"
GRAY = "#536876"
GRID = "#d8e4eb"
TEAL = "#167a80"
BLUE = "#236aa0"
ORANGE = "#bd572b"
CAT = "#70e2ec"
REMOTE = "#ffcf68"
INSTANCE_COLORS = {"cat": (CAT, "#c5a1ff"), "remote": (REMOTE, "#ff997f")}
LEFT_CAT_CROP = (0, 326, 38, 480)


def save(fig, output, name, source=None):
    options = {}
    extension = "png"
    if source:
        extension = "jpg"
        credit = f"Adapted with overlays or crops from {source['title']} by {source['credit']}. "
        credit += f"Source: {source['original']} License: {source['license']}"
        options["pil_kwargs"] = {
            "quality": 95, "subsampling": 0, "optimize": True, "comment": credit.encode("utf-8"),
        }
    path = output / f"ch23_pretrained_{name}.{extension}"
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.06, **options)
    plt.close(fig)


def photo(ax, image, title, crop=None):
    height, width = image.shape[:2]
    ax.imshow(image, extent=(0, width, height, 0), interpolation="nearest")
    left, right, top, bottom = crop or (0, width, 0, height)
    ax.set(xlim=(left, right), ylim=(bottom, top))
    ax.set_title(title, loc="left", fontweight="bold", pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def draw_box(ax, box, color, linestyle="-", label=None):
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color,
                           linewidth=1.65, linestyle=linestyle))
    if label:
        ax.annotate(label, (x0, y0), xytext=(0, 1.8), textcoords="offset points",
                    ha="left", va="bottom", fontsize=10, weight="bold", color=INK,
                    bbox={"boxstyle": "square,pad=.21", "fc": color, "ec": "none"})


def overlay(ax, mask, color, alpha=.43):
    height, width = mask.shape
    tint = np.empty((height, width, 4), dtype=np.float32)
    tint[:] = to_rgba(color)
    tint[..., 3] = mask * alpha
    ax.imshow(tint, extent=(0, width, height, 0), interpolation="nearest")


def outline(ax, mask, color):
    height, width = mask.shape
    ax.contour(np.arange(width) + .5, np.arange(height) + .5, mask,
               levels=[.5], colors=[color], linewidths=.85)


def boxes_figure(run, image, targets, output):
    fig, (reference, prediction) = plt.subplots(1, 2, figsize=(9.4, 3.9), layout="constrained")
    photo(reference, image, "(a) COCO annotations")
    photo(prediction, image, "(b) RF-DETR Nano")
    colors = {"cat": CAT, "remote": REMOTE}
    for label, rows in targets.items():
        for row in rows:
            draw_box(reference, row["box_xyxy"], colors[label], "--", label)
    for row in run["detection"]["display"]:
        draw_box(prediction, row["box_xyxy"], colors[row["label"]],
                 label=f"{row['label']}  {row['score']:.3f}")
    save(fig, output, "detection_boxes", run["image"])


def mask_errors_figure(run, image, targets, output):
    fig, (scene, detail) = plt.subplots(1, 2, figsize=(9.4, 4.2),
                                        gridspec_kw={"width_ratios": [1.62, 1]})
    photo(scene, image, "(a) RF-DETR Seg Nano masks")
    photo(detail, image, "(b) Left cat vs. COCO", LEFT_CAT_CROP)
    rows = [row for row in run["instance_segmentation"]["instances"] if row["label"] in targets]
    for row in rows:
        side = int(np.argmax([m["iou"] for m in row["mask_overlaps"]]))
        color = INSTANCE_COLORS[row["label"]][side]
        mask = decode_mask(row["mask"])
        overlay(scene, mask, color)
        outline(scene, mask, color)
    cat = next(row for row in rows if row["label"] == "cat" and row["mask_overlaps"][0]["iou"] > .5)
    predicted = decode_mask(cat["mask"])
    reference = targets["cat"][0]["mask"]
    counts = cat["mask_overlaps"][0]
    regions = [
        (reference & predicted, CAT, .40, "Shared", counts["intersection_pixels"]),
        (reference & ~predicted, REMOTE, .98, "Reference only", counts["reference_pixels"] - counts["intersection_pixels"]),
        (predicted & ~reference, "#ff76a6", .98, "Prediction only", counts["predicted_pixels"] - counts["intersection_pixels"]),
    ]
    for mask, color, alpha, _, _ in regions:
        overlay(detail, mask, color, alpha)
    handles = [Patch(facecolor=color, edgecolor=INK, linewidth=.5, label=f"{name}  {count:,}")
               for _, color, _, name, count in regions]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, columnspacing=1.9)
    fig.subplots_adjust(left=.012, right=.988, top=.92, bottom=.14, wspace=.045)
    save(fig, output, "mask_errors", run["image"])


def click_figure(run, image, output):
    click = run["prompted_segmentation"]["runs"]["foreground_click"]
    row = click["rows"][0]
    x, y = click["prompts"]["input_points"][0][0][0]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.32))
    for index, (ax, candidate) in enumerate(zip(axes, row["candidates"], strict=True)):
        tag = ""
        if index == row["selected_by_quality"]:
            tag = " · Model choice"
        elif index == row["best_measured"]:
            tag = " · Best measured IoU"
        photo(ax, image, f"{index + 1}{tag}", LEFT_CAT_CROP)
        mask = decode_mask(candidate["mask"])
        overlay(ax, mask, CAT, .52)
        outline(ax, mask, CAT)
        ax.scatter(x, y, marker="+", s=143, color=INK, linewidth=4, zorder=5)
        ax.scatter(x, y, marker="+", s=109, color="white", linewidth=1.6, zorder=6)
        ax.text(.5, -.037, f"Quality score  {candidate['quality']:.3f}\nMask IoU  {candidate['iou']:.3f}",
                transform=ax.transAxes, ha="center", va="top", linespacing=1.35)
    fig.subplots_adjust(left=.015, right=.985, top=.92, bottom=.14, wspace=.08)
    save(fig, output, "prompt_candidates", run["image"])


def box_prompt_figure(run, image, output):
    prompted = run["prompted_segmentation"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    keys = ("detector_boxes", "shifted_box")
    titles = ("(a) RF-DETR box", f"(b) Same box shifted {prompted['box_shift_x']} px right")
    for ax, key, title in zip(axes, keys, titles, strict=True):
        condition = prompted["runs"][key]
        box = condition["prompts"]["input_boxes"][0][0]
        row = condition["rows"][0]
        candidate = row["candidates"][row["selected_by_quality"]]
        photo(ax, image, title, (0, 474, 26, 480))
        mask = decode_mask(candidate["mask"])
        overlay(ax, mask, CAT, .46)
        outline(ax, mask, CAT)
        draw_box(ax, box, REMOTE, "--")
        ax.text(.5, -.044, f"Selected mask IoU  {candidate['iou']:.3f}",
                transform=ax.transAxes, ha="center", va="top", fontweight="bold")
    fig.subplots_adjust(left=.015, right=.985, top=.92, bottom=.12, wspace=.08)
    save(fig, output, "box_prompts", run["image"])


def ranking_figure(run, output):
    remote = run["detection"]["evaluation"]["remote"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.35), sharey=True)
    for ax, threshold in zip(axes, ("0.90", "0.93"), strict=True):
        result = remote[threshold]
        recall, precision = (np.asarray(result[key]) for key in ("recall", "precision"))
        matched = np.asarray(result["true_positive"])
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        for start, end, height in zip(np.r_[0, recall[:-1]], recall, envelope, strict=True):
            if end > start:
                ax.add_patch(Rectangle((start, 0), end - start, height, fc="#c8e5ed", ec=BLUE, lw=1.05))
        ax.scatter(recall[matched], precision[matched], s=59, marker="o", color=TEAL,
                   linewidths=.75, edgecolors="white", zorder=3)
        ax.scatter(recall[~matched], precision[~matched], s=32, marker="x", color=ORANGE,
                   linewidths=1.4, zorder=3)
        ax.set(xlim=(-.055, 1.055), ylim=(-.04, 1.08), xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel="Recall")
        ax.set_title(f"IoU ≥ {threshold}    ·    AP {result['ap']:.2f}", loc="left", weight="bold")
        ax.grid(color=GRID, linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Precision")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=TEAL, label="Matched at this rank"),
        Line2D([], [], marker="x", linestyle="", color=ORANGE, label="False positive at this rank"),
        Patch(facecolor="#c8e5ed", edgecolor=BLUE, label="Area under the envelope"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, columnspacing=1.5)
    fig.subplots_adjust(left=.077, right=.975, top=.88, bottom=.25, wspace=.20)
    save(fig, output, "remote_ranking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=EXAMPLE / "PRETRAINED_RUN.json")
    parser.add_argument("--media-dir", type=Path, default=Path("output/ch23-media"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch23-pretrained-figures"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11.0, "text.color": INK,
                               "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.titlecolor": INK,
                               "xtick.color": GRAY, "ytick.color": GRAY})
    run = json.loads(args.record.read_text(encoding="utf-8"))
    fixture = read_fixture()
    if run["image"] != fixture["image"]:
        raise ValueError("The run and fixture refer to different photographs")
    image = np.asarray(load_photo(fixture, args.media_dir, args.offline))
    targets = reference_instances(fixture)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boxes_figure(run, image, targets, args.output_dir)
    mask_errors_figure(run, image, targets, args.output_dir)
    click_figure(run, image, args.output_dir)
    box_prompt_figure(run, image, args.output_dir)
    ranking_figure(run, args.output_dir)
    print(f"Rendered five measured figures in {args.output_dir}")


if __name__ == "__main__":
    main()
