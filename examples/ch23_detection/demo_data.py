"""Read the pinned example photo, COCO polygons, and saved binary masks."""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from .model import mask_iou


EXAMPLE = Path(__file__).resolve().parent


def verify_file(path, expected):
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")


def load_photo(fixture, media_dir, offline):
    info = fixture["image"]
    path = Path(media_dir) / "cats.jpg"
    if not path.is_file():
        if offline:
            raise FileNotFoundError(path)
        request = Request(info["download"], headers={"User-Agent": "Transformers textbook demo/1.0"})
        with urlopen(request, timeout=60) as response:
            data = response.read(1_000_001)
        if hashlib.sha256(data).hexdigest() != info["sha256"]:
            raise ValueError("The downloaded photo differs from the pinned example")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    verify_file(path, info["sha256"])
    with Image.open(path) as source:
        photo = source.convert("RGB")
    if photo.size != (info["width"], info["height"]):
        raise ValueError("Photo dimensions differ from the annotation grid")
    return photo


def read_fixture():
    return json.loads((EXAMPLE / "COCO_39769.json").read_text(encoding="utf-8"))


def reference_instances(fixture):
    height, width = (fixture["image"][axis] for axis in ("height", "width"))
    targets = {}
    for annotation in fixture["annotations"]:
        category = fixture["categories"][str(annotation["category_id"])]
        x, y, w, h = annotation["bbox"]
        polygons = coco_mask.frPyObjects(annotation["segmentation"], height, width)
        mask = coco_mask.decode(coco_mask.merge(polygons)).astype(bool)
        targets.setdefault(category, []).append({
            "annotation_id": annotation["id"],
            "box_xyxy": [x, y, x + w, y + h],
            "mask": mask,
        })
    return targets


def encode_mask(mask):
    encoded = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": encoded["size"], "counts": encoded["counts"].decode("ascii")}


def decode_mask(encoded):
    return coco_mask.decode({**encoded, "counts": encoded["counts"].encode("ascii")}).astype(bool)


def overlap_counts(prediction, reference):
    return {
        "predicted_pixels": int(prediction.sum()),
        "reference_pixels": int(reference.sum()),
        "intersection_pixels": int(np.count_nonzero(prediction & reference)),
        "union_pixels": int(np.count_nonzero(prediction | reference)),
        "iou": mask_iou(prediction, reference),
    }
