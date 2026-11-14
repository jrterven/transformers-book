"""Photographs, captions, and paired visual questions for Chapter 24."""

from hashlib import file_digest, sha256
from pathlib import Path
from urllib.request import urlopen

from PIL import Image, ImageOps


EXAMPLE = Path(__file__).resolve().parent
PHOTO_BASE = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/"
PHOTOS = {
    "coffee": ("coffee.png", "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"),
    "chelsea": ("chelsea.png", "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb"),
    "astronaut": ("astronaut.png", "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"),
    "camera": ("camera.png", "b0793d2adda0fa6ae899c03989482bff9a42d3d5690fc7e3648f2795d730c23a"),
    "rocket": ("rocket.jpg", "c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c"),
}

# Define relevance before scoring. An empty tuple denotes a query with no match.
CAPTIONS = (
    ("espresso", "A cup of espresso on a red saucer.", ("coffee",)),
    ("cat", "A striped cat with green eyes.", ("chelsea",)),
    ("astronaut", "An astronaut in an orange spacesuit.", ("astronaut",)),
    ("tripod", "A person using a tripod outdoors.", ("camera",)),
    ("rocket", "A tall rocket on a launch pad at night.", ("rocket",)),
    ("cup_spoon", "A cup and spoon on a wooden table.", ("coffee",)),
    ("spaceflight", "A scene connected to spaceflight.", ("astronaut", "rocket")),
    ("sailboat", "A sailboat on a lake.", ()),
)

# Original and mirrored views share the question. Answers use viewer coordinates.
PAIRS = (
    ("spoon", "coffee", "On which side of the cup is the bowl of the spoon, left or right? Answer with one word.",
     ("right", "left")),
    ("flag", "astronaut", "On which side of the image is the U.S. flag, left or right? Answer with one word.",
     ("left", "right")),
    ("helmet", "astronaut", "On which side of the image is the helmet, left or right? Answer with one word.",
     ("right", "left")),
    ("handle", "coffee", "Is the cup handle on the left or the right side of the cup? Answer with one word.",
     ("left", "right")),
)

# For the first two photo pairs, exchange only the direction in the caption.
RELATION_CAPTIONS = {
    "spoon": "The bowl of the spoon is to the {} of the cup.",
    "flag": "The U.S. flag is on the {} side of the image.",
}

ATTRIBUTES = (
    ("saucer", "coffee", "What color is the saucer? Answer with one word.", ("red",)),
    ("drink", "coffee", "What drink is in the cup? Answer with one word.", ("espresso", "coffee")),
    ("suit", "astronaut", "What color is the spacesuit? Answer with one word.", ("orange",)),
    ("cat_present", "chelsea", "Is there a cat in the photo? Answer yes or no.", ("yes",)),
    ("cat_absent", "coffee", "Is there a cat in the photo? Answer yes or no.", ("no",)),
)


def verify_file(path, expected):
    with path.open("rb") as handle:
        actual = file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")


def load_photos(directory, offline):
    photos = {}
    for key, (filename, expected) in PHOTOS.items():
        path = directory / filename
        if not path.exists():
            if offline:
                raise FileNotFoundError(path)
            with urlopen(PHOTO_BASE + filename, timeout=60) as response:
                data = response.read(2_000_001)
            if sha256(data).hexdigest() != expected:
                raise ValueError(f"The downloaded photograph differs from the pinned file: {filename}")
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        verify_file(path, expected)
        with Image.open(path) as photo:
            photos[key] = photo.convert("RGB")
    return photos


def rgb_hash(photo):
    return sha256(photo.tobytes()).hexdigest()


def photo_manifest(photos):
    return {key: {"filename": filename, "sha256": digest, "size_wh": list(photos[key].size),
                  "rgb_sha256": rgb_hash(photos[key]),
                  "mirror_rgb_sha256": rgb_hash(ImageOps.mirror(photos[key]))}
            for key, (filename, digest) in PHOTOS.items()}
