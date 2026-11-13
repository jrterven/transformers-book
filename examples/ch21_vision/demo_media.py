"""Pinned public media and frame sampling for the Chapter 21 demonstrations."""

import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


ASSETS = {
    "astronaut.png": {
        "url": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png",
        "source": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut",
        "credit": "NASA, portrait of Eileen Collins, via scikit-image",
        "license": "Public domain",
        "sha256": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    },
    "fan.webm": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/60/Collections_in_Motion_Reversible_Fan.webm",
        "source": "https://commons.wikimedia.org/wiki/File:Collections_in_Motion_Reversible_Fan.webm",
        "credit": "Cooper Hewitt, Smithsonian Design Museum, Collections in Motion: Reversible Fan",
        "license": "https://creativecommons.org/licenses/by/3.0/",
        "sha256": "18228eb0f79dfef22b93437c55f4eb130c0fb9862f03d5eb14aafb35cd7dca42",
    },
}

# Windows chosen from the visible motion in one recording.
WINDOWS = (
    ("First unfolding", 36, 7, "unfolding"),
    ("Second unfolding", 744, 10, "unfolding"),
    ("Folding", 1344, 10, "folding"),
)


def verify_sha256(path, expected):
    with Path(path).open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")


def asset(directory, name, *, offline=False):
    """Read one cached source or fetch it from its fixed public URL."""
    source = ASSETS[name]
    path = Path(directory) / name
    if not path.is_file():
        if offline:
            raise FileNotFoundError(path)
        request = Request(source["url"], headers={"User-Agent": "Transformers textbook demo/1.0"})
        with urlopen(request, timeout=60) as response:
            data = response.read(8_000_001)
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ValueError(f"Downloaded {name} does not match its published demo checksum")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    verify_sha256(path, source["sha256"])
    return path


def image_pair(path, size, shift_pixels, fill):
    with Image.open(path) as source:
        original = np.array(source.convert("RGB").resize((size, size), Image.Resampling.BICUBIC))
    shifted = np.empty_like(original)
    shifted[:, :shift_pixels] = fill
    shifted[:, shift_pixels:] = original[:, :-shift_pixels]
    return original, shifted


def sampled_frames(path, indices):
    """Decode exactly the requested zero-based frame numbers in presentation order."""
    if not indices or sorted(set(indices)) != list(indices) or indices[0] < 0:
        raise ValueError("Frame indices must be nonnegative, distinct, and increasing")
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=height,width:frame=best_effort_timestamp_time",
        "-of", "json", str(path),
    ]))
    stream = info["streams"][0]
    expression = "select=" + "+".join(f"eq(n\\,{i})" for i in indices)
    data = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", expression,
        "-frames:v", str(len(indices)), "-fps_mode", "passthrough", "-an",
        "-pix_fmt", "rgb24", "-threads", "1", "-f", "rawvideo", "pipe:1",
    ])
    frames = np.frombuffer(data, dtype=np.uint8).reshape(-1, stream["height"], stream["width"], 3)
    if len(frames) != len(indices):
        raise ValueError(f"Expected {len(indices)} frames, decoded {len(frames)}")
    timestamps = [float(info["frames"][index]["best_effort_timestamp_time"]) for index in indices]
    return frames.copy(), timestamps
