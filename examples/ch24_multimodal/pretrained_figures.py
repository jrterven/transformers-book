"""Render the measured image-text rankings and visual-question comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from .demo_data import EXAMPLE, load_photos, photo_manifest


INK = "#183148"
MUTED = "#546b7a"
TEAL = "#167b77"
BLUE = "#285a99"
AMBER = "#ce742f"
GRID = "#dce7ee"
PALE = "#edf4f7"
MAP = LinearSegmentedColormap.from_list("cosine", ["#ebbe9d", "#fffdfa", "#214b78"])
GALLERY_LABELS = {"coffee": "Coffee", "chelsea": "Chelsea", "astronaut": "Astronaut", "camera": "Tripod", "rocket": "Rocket"}
CAPTION_LABELS = ("Espresso", "Striped cat", "Orange suit", "Tripod", "Launch pad", "Cup + spoon", "Spaceflight", "Sailboat")
PAIR_LABELS = {"spoon": "Spoon by the cup", "flag": "Flag in the portrait", "helmet": "Helmet in the portrait", "handle": "Cup handle"}
CREDITS = "Gallery photos from scikit-image: Rachel Michetti, Stefan van der Walt, NASA, Lav Varshney, SpaceX. See chapter guide."


def save(fig, directory, name, *, photo=False):
    suffix = "jpg" if photo else "png"
    options = {"pil_kwargs": {"quality": 93, "subsampling": 0, "optimize": True, "comment": CREDITS.encode()}} if photo else {}
    fig.savefig(directory / f"ch24_pretrained_{name}.{suffix}", dpi=166,
                facecolor="white", bbox_inches="tight", pad_inches=.06, **options)
    plt.close(fig)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def gallery_matrix(record, photos, directory):
    scores = np.array(record["siglip"]["cosines"])
    relevant = np.array(record["relevance"])
    fig = plt.figure(figsize=(10, 4.05))
    spec = fig.add_gridspec(5, 2, width_ratios=(1.25, 7.35), wspace=.025, hspace=.025)
    for row, key in enumerate(record["image_order"]):
        panel = fig.add_subplot(spec[row, 0])
        image = ImageOps.contain(photos[key], (192, 126))
        panel.imshow(image)
        bare(panel)
        panel.text(.98, .075, GALLERY_LABELS[key], transform=panel.transAxes, ha="right", va="bottom",
                   color="white", fontsize=8, weight="bold", bbox={"facecolor": INK, "edgecolor": "none", "pad": 2})

    matrix = fig.add_subplot(spec[:, 1])
    norm = TwoSlopeNorm(vmin=min(-.055, scores.min()), vcenter=0, vmax=max(.18, scores.max()))
    tile = matrix.imshow(scores, cmap=MAP, norm=norm, interpolation="nearest", aspect="auto")
    matrix.xaxis.tick_top()
    matrix.set_xticks(range(len(CAPTION_LABELS)), CAPTION_LABELS, fontsize=9.5)
    matrix.set_yticks([])
    matrix.tick_params(axis="x", length=0, pad=7)
    matrix.spines[:].set_visible(False)
    for (r, c), score in np.ndenumerate(scores):
        color = "white" if score > .114 else INK
        matrix.text(c, r, f"{score:.3f}", ha="center", va="center", fontsize=10, color=color)
        if relevant[r, c]:
            for stroke, width in (("white", 2.5), ("#102e3e", 1.05)):
                matrix.add_patch(Rectangle((c - .455, r - .43), .91, .86, fill=False, ec=stroke, lw=width))
    cbar = fig.colorbar(tile, ax=matrix, fraction=.025, pad=.019, ticks=[-.05, 0, .06, .12, .18])
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Cosine", size=10)
    fig.subplots_adjust(left=.016, right=.974, top=.88, bottom=.05)
    save(fig, directory, "gallery_matrix", photo=True)


def ranked_gallery(record, photos, directory):
    query = next(i for i, c in enumerate(record["captions"]) if c["id"] == "spaceflight")
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 4.45))
    for row, (key, title) in enumerate((("siglip", "SigLIP 2 Base"), ("embedding", "Qwen3-VL-Embedding-2B"))):
        result = record[key]
        for rank, image_index in enumerate(result["ranked"]["text_to_image"][query][:3]):
            ax = axes[row, rank]
            image_key = record["image_order"][image_index]
            image = ImageOps.pad(photos[image_key], (350, 192), method=Image.Resampling.LANCZOS, color="#eef2f5")
            ax.imshow(image)
            bare(ax)
            score = result["cosines"][image_index][query]
            is_match = record["relevance"][image_index][query]
            ax.set_title(f"{rank + 1}. {GALLERY_LABELS[image_key]}  {score:.3f}", loc="left", weight="bold",
                         color=TEAL if is_match else MUTED, fontsize=11)
            if is_match:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=TEAL, lw=2))
        axes[row, 0].text(-.01, 1.41, title, transform=axes[row, 0].transAxes, color=INK,
                         va="top", ha="left", weight="bold", fontsize=11)
    fig.subplots_adjust(left=.021, right=.98, top=.835, bottom=.025, hspace=.62, wspace=.12)
    save(fig, directory, "spaceflight_ranks", photo=True)


def tensor_trace(visual, photos, directory):
    item = visual["pairs"][0]
    original = item["original"]
    trace = original["trace"]
    _, rows, cols = trace["image_grid_thw"]
    patch = trace["patch_size"]
    width, height = cols * patch, rows * patch
    photo = photos[item["photo"]].resize((width, height), Image.Resampling.BICUBIC)
    fig, (ax, notes) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.24, 1]})
    ax.imshow(photo, extent=(0, width, height, 0))
    for c in range(patch, width, patch):
        ax.axvline(c, color="white", lw=.45, alpha=.48)
    for r in range(patch, height, patch):
        ax.axhline(r, color="white", lw=.45, alpha=.48)
    ax.add_patch(Rectangle((352, 224), 2 * patch, 2 * patch, fill=False, ec="#ffe07e", lw=2.6))
    bare(ax)
    ax.set_title(f"Resized image  ·  {width} × {height} pixels", loc="left", weight="bold", pad=9)
    notes.set_axis_off()
    entries = (
        (f"{rows} × {cols} patch positions", f"{trace['pixels'][0]} rows × {trace['pixels'][1]:,} input values"),
        ("Vision encoder", f"{trace['encoded_patches'][0]} rows × {trace['encoded_patches'][1]:,} features"),
        ("Merge 2 × 2 spatial neighbors", f"{trace['merged_tokens'][0]} rows × {trace['merged_tokens'][1]:,} features"),
        ("Language input", f"{original['image_tokens']} visual + {original['input_tokens'] - original['image_tokens']} text and format positions"),
    )
    for i, (title, detail) in enumerate(entries):
        y = .84 - .234 * i
        notes.add_patch(Rectangle((0, y - .093), 1, .185, transform=notes.transAxes, fc=PALE, ec="none"))
        notes.text(.04, y + .026, title, transform=notes.transAxes, va="center", weight="bold", fontsize=11)
        notes.text(.04, y - .043, detail, transform=notes.transAxes, va="center", color=MUTED, fontsize=10.4)
        if i < len(entries) - 1:
            notes.annotate("", xy=(.5, y - .137), xytext=(.5, y - .099), xycoords="axes fraction",
                           arrowprops={"arrowstyle": "-|>", "color": TEAL, "lw": 1.15, "shrinkA": 0, "shrinkB": 0})
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.035, wspace=.10)
    save(fig, directory, "qwen_trace", photo=True)


def answer_gallery(visual, photos, directory):
    fig, axes = plt.subplots(2, 3, figsize=(9.65, 5.05), gridspec_kw={"width_ratios": [1, 1, .75]})
    for row, pair in enumerate(visual["pairs"][:2]):
        source = photos[pair["photo"]]
        for column, (key, photo, title) in enumerate(zip(
            ("original", "mirror"), (source, ImageOps.mirror(source)), ("Original", "Mirrored"), strict=True,
        )):
            ax = axes[row, column]
            shown = ImageOps.pad(photo, (350, 215), method=Image.Resampling.LANCZOS, color="#edf1f4")
            ax.imshow(shown)
            bare(ax)
            answer = pair[key]["answer"]
            color = TEAL if pair[key]["normalized"] == pair["expected"][column] else AMBER
            ax.set_title(f"{title}   →   {answer}", loc="left", weight="bold", color=color)
        axes[row, 0].text(0, 1.45, PAIR_LABELS[pair["id"]], transform=axes[row, 0].transAxes,
                          weight="bold", fontsize=12, va="top")
        ax = axes[row, 2]
        ax.set_axis_off()
        ax.add_patch(Rectangle((0, .24), 1, .58, transform=ax.transAxes, fc=PALE, ec="none"))
        ax.text(.09, .68, "Question only", transform=ax.transAxes, weight="bold", fontsize=11)
        ax.text(.09, .48, pair["question_only"]["answer"], transform=ax.transAxes,
                weight="bold", fontsize=17, color=BLUE)
        ax.text(.09, .34, "Same answer for both", transform=ax.transAxes, color=MUTED, fontsize=9.5)
    fig.subplots_adjust(left=.016, right=.985, top=.85, bottom=.035, hspace=.68, wspace=.08)
    save(fig, directory, "vqa_pairs", photo=True)


def answer_scores(visual, directory):
    fig, (ax, count) = plt.subplots(1, 2, figsize=(9.8, 3.65), gridspec_kw={"width_ratios": [1.50, 1]})
    markers = (("original", "o", TEAL, "Original"), ("mirror", "s", BLUE, "Mirrored"),
               ("question_only", "D", AMBER, "Question only"))
    differences = [{key: pair[key]["left_right_logits"][1] - pair[key]["left_right_logits"][0]
                    for key, _, _, _ in markers} for pair in visual["pairs"]]
    limit = max(5.5, max(abs(v) for row in differences for v in row.values()) + 1)
    ax.axvspan(-limit, 0, color="#edf3f9")
    ax.axvspan(0, limit, color="#eaf5f1")
    ax.axvline(0, lw=1, color=MUTED)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(.42, 4.7)
    ax.set_yticks([4, 3, 2, 1], [PAIR_LABELS[row["id"]] for row in visual["pairs"]])
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xlabel("Right token logit minus left token logit")
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.text(-limit + .25, 4.52, "Favors left", color=BLUE, size=9)
    ax.text(limit - .25, 4.52, "Favors right", color=TEAL, size=9, ha="right")
    for y, delta in zip((4, 3, 2, 1), differences, strict=True):
        ax.plot([delta["original"], delta["mirror"]], [y, y], color="#9bb4c5", lw=1.5, zorder=2)
        for view, marker, color, _ in markers:
            ax.scatter(delta[view], y - .22 if view == "question_only" else y, color=color, s=42,
                       marker=marker, edgecolors="white", linewidths=.7, zorder=4)
    handles = [Line2D([], [], color=color, marker=marker, linestyle="", label=title)
               for _, marker, color, title in markers]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.21), ncol=3, frameon=False,
              fontsize=9, columnspacing=.85, handletextpad=.3)

    count.set_axis_off()
    count.text(.02, .96, "Decoded answers", transform=count.transAxes, va="top", weight="bold", fontsize=12)
    rows = []
    for key, label in (("visual", "Image + question"), ("text_only", "Question only")):
        m = visual["metrics"][key]
        rows.append([label, f"{m['correct_views']}/{m['views']}", f"{m['both_correct']}/{m['pairs']}"])
    table = count.table(cellText=rows, colLabels=["Input", "Views", "Pairs"], bbox=[.02, .36, .98, .46],
                        cellLoc="left", colLoc="left", colWidths=[.54, .23, .23])
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    for (i, _), cell in table.get_celld().items():
        cell.set_facecolor("#ddebf2" if i == 0 else PALE)
        cell.set_edgecolor("white")
        cell.get_text().set_color(INK)
        if i == 0:
            cell.get_text().set_weight("bold")
    count.text(.02, .25, "A pair earns credit when both views\nare answered correctly.",
               transform=count.transAxes, fontsize=10, va="top", linespacing=1.4, color=MUTED)
    fig.subplots_adjust(left=.171, right=.985, top=.93, bottom=.245, wspace=.20)
    save(fig, directory, "vqa_scores")


def relation_scores(record, directory):
    fig, axes = plt.subplots(1, 2, figsize=(9.45, 3.0))
    for ax, (key, name) in zip(axes, (("siglip", "SigLIP 2 Base"), ("embedding", "Qwen3-VL-Embedding-2B")), strict=True):
        rows = record[key]["relation_probe"]
        margins = [np.diff(row["cosines_original_mirror"], axis=1)[:, 0] for row in rows]
        limit = 1.24 * max(max(np.abs(values)) for values in margins)
        ax.axvspan(-limit, 0, color="#edf3f9")
        ax.axvspan(0, limit, color="#eaf5f1")
        ax.axvline(0, lw=1, color=MUTED)
        for center, (original, mirrored) in zip((2, 1), margins, strict=True):
            ax.plot([original, mirrored], [center + .13, center - .13], lw=1.25, color="#a6b9c8")
            ax.scatter(original, center + .13, color=TEAL, marker="o", s=58, zorder=3, edgecolors="white", linewidths=.6)
            ax.scatter(mirrored, center - .13, color=BLUE, marker="s", s=56, zorder=3, edgecolors="white", linewidths=.6)
        ax.set(xlim=(-limit, limit), ylim=(.60, 2.48), yticks=[2, 1], yticklabels=["Spoon", "Flag"])
        ax.tick_params(axis="y", length=0, pad=6)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_title(name, loc="left", weight="bold", pad=12)
        ax.text(.025, .92, "Favors left caption", color=BLUE, transform=ax.transAxes, fontsize=9)
        ax.text(.975, .92, "Favors right caption", color=TEAL, transform=ax.transAxes, ha="right", fontsize=9)
    fig.supxlabel("Right-caption cosine minus left-caption cosine", y=.20, fontsize=10.5)
    handles = [Line2D([], [], color=TEAL, marker="o", linestyle="", label="Original"),
               Line2D([], [], color=BLUE, marker="s", linestyle="", label="Mirrored")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015), ncol=2,
               frameon=False, fontsize=9.5, handletextpad=.45)
    fig.subplots_adjust(left=.079, right=.985, top=.86, bottom=.39, wspace=.25)
    save(fig, directory, "relation_ranking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=EXAMPLE / "PRETRAINED_RUN.json")
    parser.add_argument("--media-dir", type=Path, default=Path("output/ch24-media"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch24-pretrained-figures"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                               "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": GRID,
                               "xtick.color": MUTED, "ytick.color": MUTED})
    record = json.loads(args.record.read_text(encoding="utf-8"))
    photos = load_photos(args.media_dir, args.offline)
    if record["photos"] != photo_manifest(photos):
        raise ValueError("Figure photographs differ from the measured input pixels")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gallery_matrix(record, photos, args.output_dir)
    ranked_gallery(record, photos, args.output_dir)
    tensor_trace(record["visual"], photos, args.output_dir)
    answer_gallery(record["visual"], photos, args.output_dir)
    answer_scores(record["visual"], args.output_dir)
    relation_scores(record, args.output_dir)
    print(f"Rendered six measured figures in {args.output_dir}")


if __name__ == "__main__":
    main()
