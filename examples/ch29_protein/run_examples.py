"""Run Chapter 29's fixed sequence and structure examples."""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import (
    AMINO_ACIDS,
    contact_map,
    dipeptide_frequencies,
    mean_pool_residues,
    residue_composition,
)


def sequence_example():
    examples = {}
    for sequence in ("ACAC", "AACC"):
        composition = residue_composition(sequence)
        frequencies = dipeptide_frequencies(sequence)
        examples[sequence] = {
            "nonzero_composition": {
                AMINO_ACIDS[i]: float(composition[i]) for i in np.flatnonzero(composition)
            },
            "nonzero_dipeptides": {
                AMINO_ACIDS[i] + AMINO_ACIDS[j]: float(frequencies[i, j])
                for i, j in np.argwhere(frequencies > 0)
            },
        }
    return {"alphabet": AMINO_ACIDS, "sequences": examples}


def pooling_example():
    embeddings = [[9, 9], [1, 0], [1, 2], [-9, -9], [0, 0]]
    mask = [False, True, True, False, False]
    return {
        "embeddings": embeddings,
        "residue_mask": mask,
        "pooled": mean_pool_residues(embeddings, mask).tolist(),
    }


def contact_example():
    coordinates = [[0, 0, 0], [3, 0, 0], [0, 4, 0], [8, 0, 0]]
    threshold, separation = 5, 2
    contacts = contact_map(coordinates, threshold, separation)
    return {
        "coordinates": coordinates,
        "threshold": threshold,
        "min_sequence_separation": separation,
        "contact_map": contacts.tolist(),
        "unique_pairs_one_based": (np.argwhere(np.triu(contacts, k=1)) + 1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "sequence_features": sequence_example(),
        "pooled_residues": pooling_example(),
        "contacts": contact_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
