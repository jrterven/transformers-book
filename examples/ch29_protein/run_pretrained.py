"""Run ESMC and ESMFold2-Fast on the chapter's fixed PDB examples."""

import argparse
from hashlib import file_digest
from importlib.metadata import version
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .model import (
    ca_lddt, contact_map, cosine_matrix,
    mean_pool_residues, residue_composition, superpose_coordinates,
)
from .reference_data import EMBEDDING_TARGETS, FOLD_TARGETS, load_references


ESMC = "biohub/ESMC-300M"
ESMFOLD = "biohub/ESMFold2-Fast"
CHECKPOINTS = {
    ESMC: (
        "96a2d8d61654ac0c08881be44da6c867686449b9",
        {"model.safetensors":
         "b06a03aa359c29a889151da4a07a517f2c307c1fcc61c8b767ec5360dd40cb49"},
    ),
    ESMFOLD: (
        "45fe8656f5b3ef493c17fcf9abe9a2968902e712",
        {
            "model-00001-of-00006.safetensors": "2bbc3d3b6d48dd39a698ef5bf8f1bea659ce8b19f4444145503a4785af962618",
            "model-00002-of-00006.safetensors": "999157d7d20926c54f2ee02c4ee7e46eddc16bbf8963850e6b59d718b36afa59",
            "model-00003-of-00006.safetensors": "d8a3d8121d4c0d41d9fdac2b7d093048a7d54b569840f6f639eb060e7ebd9211",
            "model-00004-of-00006.safetensors": "f28625f5d104e7ce0fafbdb2eacedb53c4e5fbfb2fb080f26995e81f0858348b",
            "model-00005-of-00006.safetensors": "2e15b21fcedab793eebc39301bb608765023824a54a5dc6b748b6e7e0d946203",
            "model-00006-of-00006.safetensors": "34fc8868f3af8a0bf3e624996eb451906054889b3e7247b03c5ac2f9f838d8da",
        },
    ),
}
SEED = 2900
FOLD_OPTIONS = {"num_loops": 3, "num_sampling_steps": 14, "num_diffusion_samples": 1}
CONTACT_DISTANCE = 8
CONTACT_SEPARATION = 6


def checkpoint_directory(repo, local_dir):
    from huggingface_hub import snapshot_download

    revision, checksums = CHECKPOINTS[repo]
    if local_dir is None:
        path = Path(snapshot_download(
            repo, revision=revision,
            allow_patterns=["config.json", "tokenizer*.json", "special_tokens_map.json",
                            "*.safetensors", "*.safetensors.index.json"],
        ))
    else:
        path = local_dir / repo.split("/")[1]
    for name, expected in checksums.items():
        with (path / name).open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"checkpoint hash mismatch for {repo}/{name}")
    return path


def embedding_demo(refs, checkpoint):
    import torch
    from transformers import AutoTokenizer, EsmcModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    encoder, info = EsmcModel.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.float32, output_loading_info=True
    )
    unexpected = {key for key in info["unexpected_keys"] if not key.startswith("lm_head.")}
    if info["missing_keys"] or info["mismatched_keys"] or unexpected:
        raise ValueError(f"ESMC encoder weights did not load: {info}")
    encoder.eval()
    sequences = [refs[pdb_id]["sequence"] for pdb_id in EMBEDDING_TARGETS]
    batch = tokenizer(
        sequences, padding=True, return_tensors="pt", return_special_tokens_mask=True
    )
    special = batch.pop("special_tokens_mask").bool()
    residue_mask = (batch["attention_mask"].bool() & ~special).numpy()
    with torch.inference_mode():
        hidden = encoder(**batch).last_hidden_state.numpy()
    counts = residue_mask.sum(axis=1)
    if counts.tolist() != [len(seq) for seq in sequences]:
        raise ValueError("tokenization must select exactly one row per input residue")
    pooled = np.stack([
        mean_pool_residues(rows, mask) for rows, mask in zip(hidden, residue_mask, strict=True)
    ])
    composition = np.stack([residue_composition(seq) for seq in sequences])
    comparison = {"composition_cosine": cosine_matrix(composition).tolist(),
                  "embedding_cosine": cosine_matrix(pooled).tolist()}

    # Two identical letters in one chain provide a direct contextual example.
    sequence = sequences[0]
    positions = [i for i, residue in enumerate(sequence, start=1) if residue == "L"][:2]
    example = hidden[0, residue_mask[0]][np.array(positions) - 1]
    return {
        "targets": EMBEDDING_TARGETS, "batched_shape": hidden.shape,
        "residue_counts": counts.tolist(), "pooled_vectors": pooled.tolist(),
        "comparison": comparison,
        "same_letter": {
            "pdb_id": EMBEDDING_TARGETS[0], "letter": "L", "positions": positions,
            "embedding_cosine": float(cosine_matrix(example)[0, 1]),
        },
    }


def score_structure(predicted, ref):
    observed = np.asarray(ref["ca"])
    positions = np.asarray(ref["positions"])
    mobile = predicted[positions - 1]
    aligned = superpose_coordinates(mobile, observed)
    offsets = np.linalg.norm(aligned - observed, axis=1)
    lddt, local_scores = ca_lddt(mobile, observed)
    eligible = np.triu(positions[None, :] - positions[:, None] >= CONTACT_SEPARATION, k=1)
    predicted_contact = contact_map(mobile, CONTACT_DISTANCE)[eligible]
    reference_contact = contact_map(observed, CONTACT_DISTANCE)[eligible]
    tp = int(np.sum(predicted_contact & reference_contact))
    fp = int(np.sum(predicted_contact & ~reference_contact))
    fn = int(np.sum(~predicted_contact & reference_contact))
    return {
        "positions": positions.tolist(), "rmsd": float(np.sqrt(np.mean(offsets**2))),
        "lddt_ca": lddt, "per_residue_lddt_ca": local_scores.tolist(),
        "offset_angstrom": offsets.tolist(),
        "contacts": {"tp": tp, "fp": fp, "fn": fn},
    }


def fold_chain(model, sequence, seed):
    import torch
    from transformers.models.esmfold2.protein_utils import prepare_protein_features

    torch.manual_seed(seed)
    features = prepare_protein_features(sequence, device="cpu")
    # The generic featurizer builds a one-row MSA even for the Fast checkpoint.
    # With these keys omitted, the native trunk skips the untrained MSA layers.
    for key in ("msa", "msa_attention_mask", "has_deletion", "deletion_value", "deletion_mean"):
        features.pop(key)
    with torch.inference_mode():
        result = model.fold(**features, **FOLD_OPTIONS)

    # Select the actual CA atoms by name, one per residue in sequence order.
    ca_code = torch.tensor([ord("C") - 32, ord("A") - 32, 0, 0])
    mask = features["atom_attention_mask"][0] & (features["ref_atom_name_chars"][0] == ca_code).all(-1)
    if not torch.equal(features["atom_to_token"][0, mask], torch.arange(len(sequence))):
        raise ValueError("expected one C-alpha atom per input residue in order")
    coords = result.sample_atom_coords[0, mask].float().numpy()
    confidence = result.plddt_per_atom[0, mask].float().numpy() * 100
    pae = result.pae[0].float().numpy()
    if (coords.shape != (len(sequence), 3) or not np.isfinite(coords).all()
            or not np.all((0 <= confidence) & (confidence <= 100))
            or not np.isfinite(pae).all() or pae.shape != (len(sequence), len(sequence))):
        raise ValueError("invalid structure or confidence from ESMFold2")
    ptm = float(result.ptm[0])
    if not 0 <= ptm <= 1:
        raise ValueError("invalid predicted TM-score")
    return coords, confidence, pae, ptm


def fold_demo(refs, checkpoint):
    import torch
    from transformers import EsmFold2Model

    model, info = EsmFold2Model.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16, output_loading_info=True
    )
    if model.config.msa_encoder.enabled:
        raise ValueError("expected the single-sequence Fast checkpoint")
    # The native class also instantiates the optional MSA branch for the Fast config.
    missing = set(info["missing_keys"])
    missing = {key for key in missing if not key.startswith("msa_encoder.")}
    if missing or info["mismatched_keys"] or info["unexpected_keys"]:
        raise ValueError(f"ESMFold2 weights did not load: {info}")
    model.eval()
    structures = {}
    for pdb_id in FOLD_TARGETS:
        sequence = refs[pdb_id]["sequence"]
        print(f"Predicting {pdb_id} ({len(sequence)} residues)", flush=True)
        coords, confidence, pae, ptm = fold_chain(model, sequence, SEED)
        comparison = {pdb_id: score_structure(coords, refs[pdb_id])}
        record = {
            "ca": coords.tolist(), "plddt_ca": confidence.tolist(),
            "ptm": ptm, "comparisons": comparison,
        }
        if pdb_id == "1CLL":
            alternative = refs["1CFD"]
            if alternative["sequence"] != sequence:
                raise ValueError("the two calmodulin references must have the same sequence")
            shared = refs[pdb_id]["positions"]
            ca_by_position = dict(zip(alternative["positions"], alternative["ca"], strict=True))
            matched = dict(alternative, positions=shared, ca=[ca_by_position[p] for p in shared])
            comparison["1CFD"] = score_structure(coords, matched)
            record["pae"] = np.round(pae.astype(np.float64), 4).tolist()
        structures[pdb_id] = record
    return structures


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", choices=("embeddings", "structures"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    refs = load_references()
    repo = ESMC if args.demo == "embeddings" else ESMFOLD
    checkpoint = checkpoint_directory(repo, args.checkpoints)
    revision, hashes = CHECKPOINTS[repo]
    report = {
        "repo": repo, "revision": revision, "sha256": hashes,
        "device": "cpu", "dtype": "float32" if args.demo == "embeddings" else "bfloat16",
        "versions": {name: version(name) for name in ("torch", "transformers", "numpy")},
        "threads": args.threads,
    }
    if args.demo == "embeddings":
        report["result"] = embedding_demo(refs, checkpoint)
    else:
        report.update(seed=SEED, options=FOLD_OPTIONS, contact_distance=CONTACT_DISTANCE,
                      contact_separation=CONTACT_SEPARATION)
        report["result"] = fold_demo(refs, checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
    with NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as stream:
        stream.write(content)
    try:
        os.replace(stream.name, args.output)
    finally:
        Path(stream.name).unlink(missing_ok=True)
    print(args.output)


if __name__ == "__main__":
    main()
