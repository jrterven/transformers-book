"""Read the fixed Protein Data Bank examples for Chapter 29."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

from .model import AMINO_ACIDS


DATA = Path(__file__).with_name("data") / "pdb_references.json"
ENTRIES = {
    "1UBQ": ("A", "Human ubiquitin", "ubiquitin",
             "056f98710cb2b36f633c45e41902a02eb446e82871da21ff2dd44f74a56ca0f6"),
    "2G3Q": ("B", "Yeast ubiquitin", "ubiquitin",
             "5019d4f2f635b81df7885e081a589494b14552bb3ff24281c26542fd2a8122a8"),
    "1U7S": ("A", "Whale myoglobin", "myoglobin",
             "79e3e6bddd9c0737ea8b1576c3bd4abea3e86de697e4aede487ef92805a37eec"),
    "1WLA": ("A", "Horse myoglobin", "myoglobin",
             "fd1e45596aa5c01f9df51fc5861a2d92ad8bfea76da138bcc3da13364185d6d9"),
    "1YU5": ("X", "Chicken villin headpiece", "villin",
             "6ab7d82d7e6a4dd0fe9adb06d5cfb399e8714b7d171afd409a8ca371ab698735"),
    "1CLL": ("A", "Calmodulin with calcium", "calmodulin",
             "a8be375d00d2d5d47ffacae491f775167d1c0488ab8485cabd269410e596d3aa"),
    "1CFD": ("A", "Calcium-free calmodulin", "calmodulin",
             "6a735efbe235cf42b4c37290bb65f57fe6d970ac15278e60949347e5ae76a83e"),
}
FOLD_TARGETS = ("1UBQ", "1YU5", "1CLL")
EMBEDDING_TARGETS = ("1UBQ", "2G3Q", "1U7S", "1WLA", "1YU5", "1CLL")
STRUCTURAL_REFERENCES = ("1UBQ", "1YU5", "1CLL", "1CFD")


def load_references():
    return json.loads(DATA.read_text(encoding="utf-8"))


def extract_reference(pdb_id, directory):
    """Extract a deposited sequence and, where needed, measured C-alpha positions."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.SeqUtils import seq1

    chain, name, family, expected = ENTRIES[pdb_id]
    path = directory / f"{pdb_id}.cif"
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"the mmCIF snapshot does not match for {pdb_id}")
    raw = MMCIF2Dict(str(path))
    entity, = [i for i, ids in enumerate(raw["_entity_poly.pdbx_strand_id"])
               if chain in ids.replace(" ", "").split(",")]
    sequence = "".join(raw["_entity_poly.pdbx_seq_one_letter_code_can"][entity].split())
    if not sequence or set(sequence) - set(AMINO_ACIDS):
        raise ValueError(f"{pdb_id} does not contain the expected standard-residue sequence")

    result = {
        "name": name, "family": family, "pdb_id": pdb_id, "auth_chain": chain,
        "sequence": sequence, "method": raw["_exptl.method"][0], "source_sha256": expected,
        "url": f"https://www.rcsb.org/structure/{pdb_id}",
        "pdb_revision": ".".join(raw[f"_pdbx_audit_revision_history.{field}"][-1]
                                 for field in ("major_revision", "minor_revision")),
    }
    resolution = raw.get("_refine.ls_d_res_high")
    if resolution:
        result["resolution_angstrom"] = float(resolution[0])
    if pdb_id not in STRUCTURAL_REFERENCES:
        return result

    columns = (
        "group_PDB", "auth_asym_id", "label_atom_id", "pdbx_PDB_model_num",
        "label_seq_id", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z",
    )
    coordinates = {}
    for kind, auth, atom, model, seq_id, residue, x, y, z in zip(
        *(raw[f"_atom_site.{column}"] for column in columns), strict=True
    ):
        if (kind, auth, atom, model) != ("ATOM", chain, "CA", "1"):
            continue
        position = int(seq_id)
        if position in coordinates or sequence[position - 1] != seq1(residue):
            raise ValueError(f"ambiguous C-alpha mapping in {pdb_id} at {position}")
        coordinates[position] = [float(x), float(y), float(z)]
    positions = sorted(coordinates)
    result.update(positions=positions, ca=[coordinates[i] for i in positions], model_number=1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmcif-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refs = {pdb_id: extract_reference(pdb_id, args.mmcif_dir) for pdb_id in ENTRIES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(refs, separators=(",", ":"), allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
