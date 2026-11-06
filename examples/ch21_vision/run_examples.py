"""Run the Chapter 21 examples without model or data downloads."""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import (
    patch_embeddings,
    patchify,
    probe_metrics,
    space_time_pairs,
    train_linear_probe,
)


TRAIN_PROFILES = (
    (0.05, 0.60, 0), (0.10, 0.75, 1), (0.20, 0.80, 0),
    (0.25, 0.95, 1), (0.05, 0.85, 1), (0.30, 0.90, 0),
)
TEST_PROFILES = (
    (0.12, 0.72, 0), (0.18, 0.88, 1), (0.26, 0.76, 0), (0.08, 0.92, 1),
)


def patch_example():
    image = np.arange(8 * 8 * 3).reshape(8, 8, 3)
    patches = patchify(image, 4)
    weight = np.zeros((48, 6))
    weight[:6] = np.eye(6)
    cls = [1, 0, 0, 0, 0, 0]
    positions = np.repeat(0.1 * np.arange(5)[:, None], 6, axis=1)
    tokens = patch_embeddings(patches, weight, cls=cls, positions=positions)
    return {
        "image_shape": list(image.shape),
        "patch_shape": list(patches.shape),
        "projection_shape": list(weight.shape),
        "token_shape_with_cls": list(tokens.shape),
        "first_six_values_per_patch": patches[:, :6].tolist(),
        "cls_token": tokens[0].tolist(),
        "tokens": tokens.tolist(),
    }


def profile_images(profiles):
    """Make one upper and one lower bright patch at each profile."""
    images, labels = [], []
    for background, bright, column in profiles:
        for label in (0, 1):
            image = np.full((8, 8, 3), background)
            row = 4 * label
            col = 4 * column
            image[row:row + 4, col:col + 4] = bright
            images.append(image)
            labels.append(label)
    return np.stack(images), np.array(labels)


def patch_mean_features(images):
    return np.stack([patchify(image, 4).mean(axis=1) for image in images])


def probe_example():
    train_images, train_labels = profile_images(TRAIN_PROFILES)
    test_images, test_labels = profile_images(TEST_PROFILES)
    train = patch_mean_features(train_images)
    test = patch_mean_features(test_images)
    results = {}
    for name, x_train, x_test in (
        ("ordered", train, test),
        ("orderless_sorted", np.sort(train, axis=1), np.sort(test, axis=1)),
    ):
        weight, bias = train_linear_probe(x_train, train_labels)
        results[name] = {
            "parameters": weight.size + bias.size,
            "train": probe_metrics(x_train, train_labels, weight, bias),
            "held_out": probe_metrics(x_test, test_labels, weight, bias),
        }
    return {
        "train_images": len(train),
        "held_out_images": len(test),
        "feature_width": train.shape[1],
        **results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "patches": patch_example(),
        "linear_probe": probe_example(),
        "video_pairs": space_time_pairs(3, 4),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
