"""Small NumPy examples for image and video representations."""

import math
from operator import index

import numpy as np

from examples.ch04_attention.attention import stable_softmax


# snippet:patchify
def patchify(image, patch_size):
    """Flatten an HWC image into nonoverlapping patches in row-major order."""
    image = np.asarray(image)
    patch_size = index(patch_size)
    if image.ndim != 3 or min(image.shape) < 1:
        raise ValueError("image needs nonempty shape [height, width, channels]")
    if image.dtype.kind not in "iuf" or not np.isfinite(image).all():
        raise ValueError("image pixels must be finite real numbers")
    height, width, channels = image.shape
    if patch_size < 1 or height % patch_size or width % patch_size:
        raise ValueError("a positive patch size must divide both image dimensions")

    rows, cols = height // patch_size, width // patch_size
    grid = image.reshape(rows, patch_size, cols, patch_size, channels)
    grid = grid.transpose(0, 2, 1, 3, 4)
    return grid.reshape(rows * cols, patch_size**2 * channels)
# end-snippet:patchify


def _real_array(array, name):
    array = np.asarray(array)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


# snippet:patch_embeddings
def patch_embeddings(patches, weight, *, cls=None, positions=None):
    """Project every patch, then optionally add a CLS row and positions."""
    patches = _real_array(patches, "patches")
    weight = _real_array(weight, "weight")
    if (patches.ndim != 2 or weight.ndim != 2 or min(patches.shape) < 1
            or min(weight.shape) < 1 or patches.shape[1] != weight.shape[0]):
        raise ValueError("patch and projection widths must match")

    with np.errstate(over="raise", invalid="raise"):
        tokens = patches @ weight
        if cls is not None:
            cls = _real_array(cls, "CLS")
            if cls.shape != (tokens.shape[1],):
                raise ValueError("CLS needs one vector at the model width")
            tokens = np.concatenate((cls[None, :], tokens), axis=0)
        if positions is not None:
            positions = _real_array(positions, "positions")
            if positions.shape != tokens.shape:
                raise ValueError("provide one positional vector per token")
            tokens = tokens + positions
    return tokens
# end-snippet:patch_embeddings


def _probe_inputs(features, labels):
    features = _real_array(features, "features")
    labels = np.asarray(labels)
    if features.ndim != 2 or min(features.shape) < 1:
        raise ValueError("features need nonempty shape [samples, width]")
    if labels.shape != (len(features),) or labels.dtype.kind not in "iub":
        raise ValueError("provide one binary integer label per sample")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("labels must be 0 or 1")
    return features, labels.astype(np.intp, copy=False)


# snippet:linear_probe
def train_linear_probe(features, labels, *, steps=120, lr=0.5):
    """Fit only a two-class affine head on fixed feature rows."""
    features, labels = _probe_inputs(features, labels)
    steps = index(steps)
    if steps < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("steps and learning rate must be positive")

    weight = np.zeros((features.shape[1], 2))
    bias = np.zeros(2)
    target = np.eye(2)[labels]
    with np.errstate(over="raise", invalid="raise"):
        for _ in range(steps):
            probabilities = stable_softmax(features @ weight + bias)
            difference = (probabilities - target) / len(features)
            weight -= lr * (features.T @ difference)
            bias -= lr * difference.sum(axis=0)
    return weight, bias
# end-snippet:linear_probe


def probe_metrics(features, labels, weight, bias):
    """Measure cross-entropy and accuracy without updating the head."""
    features, labels = _probe_inputs(features, labels)
    weight = _real_array(weight, "weight")
    bias = _real_array(bias, "bias")
    if weight.shape != (features.shape[1], 2) or bias.shape != (2,):
        raise ValueError("probe weight and bias must match width and two classes")
    with np.errstate(over="raise", invalid="raise"):
        logits = features @ weight + bias
        rows = np.arange(len(labels))
        wrong_minus_right = logits[rows, 1 - labels] - logits[rows, labels]
        loss = np.logaddexp(0, wrong_minus_right).mean()
    return {"loss": float(loss), "accuracy": float((logits.argmax(axis=1) == labels).mean())}


# snippet:space_time_pairs
def space_time_pairs(frames, patches_per_frame):
    """Count ordered attention pairs, including self-pairs, with no CLS row."""
    frames, patches_per_frame = map(index, (frames, patches_per_frame))
    if min(frames, patches_per_frame) < 1:
        raise ValueError("frame and patch counts must be positive")
    spatial = frames * patches_per_frame**2
    temporal = patches_per_frame * frames**2
    return {
        "joint": (frames * patches_per_frame) ** 2,
        "spatial": spatial,
        "temporal": temporal,
        "divided": spatial + temporal,
    }
# end-snippet:space_time_pairs
