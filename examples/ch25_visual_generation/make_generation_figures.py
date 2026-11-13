"""Lay out chapter figures from the generated PNGs without altering their content."""

import argparse
import json
from pathlib import Path

from matplotlib import get_data_path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


INK = "#243b4c"
SECONDARY = "#445b70"
BACKGROUND = "#f3f5f7"
BORDER = "#a8b6c1"
FONT_DIR = Path(get_data_path()) / "fonts" / "ttf"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def panel_grid(rows, column_titles, row_titles, tile_size, path):
    tile_width, tile_height = tile_size
    margin, gutter = 18, 14
    heading = 53 if column_titles else 0
    row_heading = 51 if row_titles else 0
    width = 2 * margin + len(rows[0]) * (tile_width + gutter) - gutter
    height = 2 * margin + heading + len(rows) * (tile_height + row_heading + gutter) - gutter
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(column_titles):
        left = margin + col * (tile_width + gutter)
        draw.text((left + tile_width / 2, margin + heading / 2 - 4), label,
                  font=font(32, bold=True), fill=INK, anchor="mm")
    for row, images in enumerate(rows):
        top = margin + heading + row * (tile_height + row_heading + gutter)
        if row_titles:
            draw.text((margin + 2, top + row_heading / 2 - 4), row_titles[row],
                      font=font(31, bold=True), fill=SECONDARY, anchor="lm")
        top += row_heading
        for col, path_image in enumerate(images):
            left = margin + col * (tile_width + gutter)
            with Image.open(path_image) as original:
                # Round the tile height to the nearest pixel to keep the source aspect ratio.
                assert abs(original.height * tile_width / original.width - tile_height) <= 0.5
                tile = original.convert("RGB").resize(tile_size, Image.Resampling.LANCZOS)
            draw.rectangle((left - 1, top - 1, left + tile_width, top + tile_height), outline=BORDER)
            canvas.paste(tile, (left, top))
    canvas.save(path, quality=92, subsampling=0, optimize=True)


def image_figures(images, output):
    report = json.loads((images / "image-run.json").read_text())
    cases = {(case["seed"], case["color"]): case for case in report["cases"]}
    seeds = sorted({seed for seed, color in cases})
    grid = [[images / cases[seed, color]["file"] for color in ("red", "blue")] for seed in seeds]
    panel_grid(grid, ["Prompt says red", "Prompt says blue"],
               [f"Seed {seed}" for seed in seeds], (640, 384), output / "ch25_flux_color.jpg")

    case = cases[2501, "red"]
    steps = report["steps"]
    states = [images / "image-states" / f"seed_2501_step{i}of{steps}.png" for i in range(steps + 1)]
    labels = [f"{i}  |  {sigma:.3f}" for i, sigma in enumerate(case["sigmas"])]
    panel_grid([states], labels, [], (256, 154), output / "ch25_flux_steps.jpg")


def rgb(path):
    with Image.open(path) as frame:
        return np.asarray(frame.convert("RGB"), dtype=np.int16)


def red_centroid(frame):
    """Return a color-only horizontal centroid and the selected pixel fraction."""
    red, green, blue = np.moveaxis(frame, -1, 0)
    selected = (red > 100) & (red - green > 50) & (red - blue > 45)
    _, x = np.nonzero(selected)
    center = float(x.mean() / (frame.shape[1] - 1)) if x.size else None
    return {"x_fraction": center, "red_fraction": float(selected.mean())}


def video_figures(images, wan, lingbot, output):
    metrics = {}
    if wan:
        indices = (0, 16, 32, 48)
        directories = [wan / f"wan-{mode}-frames" for mode in ("text", "image")]
        grid = [[directory / f"{i:03d}.png" for i in indices] for directory in directories]
        panel_grid(grid, [f"{i / 16:g} s" for i in indices],
                   ["Text only", "Text + first-frame image"], (320, 192), output / "ch25_wan_frames.jpg")
        source, first = rgb(images / "train-red-2501.png"), rgb(grid[1][0])
        panel_grid([[images / "train-red-2501.png", grid[1][0]]],
                   ["FLUX.2 reference", "Wan decoded frame 0"], [], (640, 384),
                   output / "ch25_wan_reference.jpg")
        metrics["reference_mean_absolute_rgb_difference_8bit"] = float(np.abs(first - source).mean())
        for directory in directories:
            metrics[directory.name] = [
                {"frame": i, "seconds": i / 16, **red_centroid(rgb(path))}
                for i, path in enumerate(sorted(directory.glob("*.png")))
            ]
    if lingbot:
        indices = (0, 8, 16, 24)
        directory = lingbot / "lingbot-text-frames"
        paths = [directory / f"{i:03d}.png" for i in indices]
        panel_grid([paths], [f"{i / 16:g} s" for i in indices], [], (320, 192),
                   output / "ch25_lingbot_frames.jpg")
        metrics[directory.name] = [
            {"frame": i, "seconds": i / 16, **red_centroid(rgb(path))}
            for i, path in enumerate(sorted(directory.glob("*.png")))
        ]
    if metrics:
        (output / "video-observations.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--wan-dir", type=Path)
    parser.add_argument("--lingbot-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_figures(args.image_dir, args.output_dir)
    video_figures(args.image_dir, args.wan_dir, args.lingbot_dir, args.output_dir)
