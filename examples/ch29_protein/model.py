"""Offline NumPy teaching examples for protein sequences and geometry."""

import math
from operator import index

import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
RESIDUE_INDEX = {residue: i for i, residue in enumerate(AMINO_ACIDS)}


def _residue_ids(sequence):
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("provide a nonempty sequence of uppercase residues")
    try:
        return np.array([RESIDUE_INDEX[residue] for residue in sequence], dtype=np.intp)
    except KeyError as error:
        raise ValueError("use only the 20 standard uppercase residue symbols") from error


# snippet:residue_composition
def residue_composition(sequence):
    """Return residue fractions in AMINO_ACIDS order."""
    ids = _residue_ids(sequence)
    return np.bincount(ids, minlength=len(AMINO_ACIDS)) / len(ids)
# end-snippet:residue_composition


# snippet:dipeptide_frequencies
def dipeptide_frequencies(sequence):
    """Return adjacent ordered-pair frequencies, or zeros for one residue."""
    ids = _residue_ids(sequence)
    width = len(AMINO_ACIDS)
    if len(ids) == 1:
        return np.zeros((width, width))
    pair_ids = width * ids[:-1] + ids[1:]
    counts = np.bincount(pair_ids, minlength=width**2).reshape(width, width)
    return counts / len(pair_ids)
# end-snippet:dipeptide_frequencies


# snippet:mean_pool_residues
def mean_pool_residues(embeddings, residue_mask):
    """Average only the embedding rows marked as residues."""
    embeddings = np.asarray(embeddings)
    residue_mask = np.asarray(residue_mask)
    if (embeddings.ndim != 2 or min(embeddings.shape) == 0
            or embeddings.dtype.kind not in "iuf"):
        raise ValueError("embeddings need a nonempty real [positions, width] matrix")
    if residue_mask.shape != embeddings.shape[:1] or residue_mask.dtype != np.bool_:
        raise ValueError("provide one Boolean mask value per embedding row")
    if not residue_mask.any():
        raise ValueError("select at least one residue row")
    with np.errstate(over="raise", invalid="raise"):
        residues = embeddings[residue_mask].astype(np.float64, copy=False)
        if not np.isfinite(residues).all():
            raise ValueError("selected residue embeddings must be finite")
        return residues.mean(axis=0)
# end-snippet:mean_pool_residues


def _contact_inputs(coordinates, threshold, min_sequence_separation):
    coordinates = np.asarray(coordinates)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 3
            or coordinates.dtype.kind not in "iuf"):
        raise ValueError("coordinates need a real [residues, 3] matrix")
    with np.errstate(over="raise", invalid="raise"):
        coordinates = coordinates.astype(np.float64, copy=False)
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    threshold = np.asarray(threshold)
    if threshold.ndim != 0 or threshold.dtype.kind not in "iuf":
        raise ValueError("threshold must be a nonnegative finite scalar")
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a nonnegative finite scalar")
    min_sequence_separation = index(min_sequence_separation)
    if min_sequence_separation < 1:
        raise ValueError("minimum sequence separation must be at least one")
    return coordinates, threshold, min_sequence_separation


# snippet:contact_map
def contact_map(coordinates, threshold, min_sequence_separation=1):
    """Return a symmetric Boolean matrix of contacts under two cutoffs."""
    coordinates, threshold, min_sequence_separation = _contact_inputs(
        coordinates, threshold, min_sequence_separation
    )
    # Scale each pair and its threshold together before squaring.
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        difference = coordinates[:, None, :] - coordinates[None, :, :]
        largest = np.maximum(np.max(np.abs(difference), axis=-1), threshold)
        _, exponent = np.frexp(largest)
        scaled = np.ldexp(difference, -exponent[..., None])
        cutoff = np.ldexp(threshold, -exponent)
        within = np.isfinite(largest) & ((scaled**2).sum(axis=-1) <= cutoff**2)
    indices = np.arange(len(coordinates))
    separation = np.abs(indices[:, None] - indices[None, :])
    return within & (separation >= min_sequence_separation)
# end-snippet:contact_map
