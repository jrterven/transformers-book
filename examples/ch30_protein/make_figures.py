"""Render Chapter 30 figures from the saved model outputs and PDB references."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from .model import contact_map, superpose_coordinates
from .reference_data import load_references
from .run_pretrained import CONTACT_DISTANCE, CONTACT_SEPARATION


RECORDS = Path(__file__).with_name("records")
INK, MUTED, GRID = "#223a4f", "#536675", "#dce6ee"
BLUE, ORANGE, GREEN = "#24638e", "#b35d32", "#247356"
CONTACT_COLORS = ["#f8fafb", ORANGE, BLUE, INK]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, output, filename):
    fig.savefig(output / filename, bbox_inches="tight", pad_inches=0.07,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def plot_embeddings(report, output):
    fig, axes = plt.subplots(1, 2, figsize=(8.45, 4), layout="constrained")
    targets = report["targets"]
    settings = (
        ("composition_cosine", "(a) Amino-acid composition"),
        ("embedding_cosine", "(b) ESMC-300M embeddings"),
    )
    for ax, (field, title) in zip(axes, settings, strict=True):
        data = np.array(report["comparison"][field])
        hm = ax.imshow(data, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        color="white" if value > 0.56 else INK, fontsize=9.5)
        for i, j in ((0, 1), (1, 0), (2, 3), (3, 2)):
            ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94,
                                   fill=False, edgecolor="#f0af7a", linewidth=1.6))
        ax.set_xticks(np.arange(len(targets)), targets)
        ax.set_yticks(np.arange(len(targets)), targets)
        ax.set(title=title)
        ax.tick_params(length=0, labelsize=9)
    colorbar = fig.colorbar(hm, ax=axes, fraction=0.032, shrink=0.89)
    colorbar.set_label("Cosine similarity")
    save(fig, output, "ch30_embedding_similarity.pdf")


def view_basis(points, tilt, angle):
    """Use a fixed projection in the principal-axis frame of the measured points."""
    _, _, right_t = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    basis = right_t.T
    # Fix signs from the first point on the two longest axes.
    basis[:, :2] *= np.sign((points[0] - points.mean(axis=0)) @ basis[:, :2])
    basis[:, 2] = np.cross(basis[:, 0], basis[:, 1])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    turn = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    pitch = np.array([[1, 0, 0], [0, cos_t, -sin_t], [0, sin_t, cos_t]])
    return (basis @ turn @ pitch)[:, :2]


def projected_traces(prediction, ref, tilt, angle):
    positions = np.asarray(prediction["comparisons"][ref["pdb_id"]]["positions"])
    ref_positions = np.asarray(ref["positions"])
    indices = np.searchsorted(ref_positions, positions)
    if not np.array_equal(ref_positions[indices], positions):
        raise ValueError("the plot requires the same residues as the evaluation")
    coords = np.asarray(ref["ca"])[indices]
    aligned = superpose_coordinates(np.asarray(prediction["ca"])[positions - 1], coords)
    basis = view_basis(coords, tilt, angle)
    center = coords.mean(axis=0)
    return (coords - center) @ basis, (aligned - center) @ basis


def draw_trace(ax, trace, color, *, linestyle="-", linewidth=1.7, zorder=3):
    ax.plot(trace[:, 0], trace[:, 1], color=color, linestyle=linestyle,
            linewidth=linewidth, solid_capstyle="round", zorder=zorder)


def plot_overlays(report, refs, output):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    fig, axes = plt.subplots(1, 2, figsize=(8.45, 3.45), layout="constrained")
    settings = (("1UBQ", "Ubiquitin", 0.30, -0.20), ("1YU5", "Villin headpiece", -0.28, 0.26))
    for i, (ax, (pdb_id, label, tilt, angle)) in enumerate(zip(axes, settings, strict=True)):
        record = report[pdb_id]
        reference, prediction = projected_traces(record, refs[pdb_id], tilt, angle)
        draw_trace(ax, reference, BLUE, linewidth=2.25)
        draw_trace(ax, prediction, ORANGE, linestyle="--", linewidth=1.6, zorder=4)
        ax.scatter(reference[[0, -1], 0], reference[[0, -1], 1], color=BLUE, s=22, zorder=5)
        for point, label_end in zip(reference[[0, -1]], ("N", "C"), strict=True):
            horizontal = -7 if label_end == "C" and pdb_id == "1YU5" else 6
            ax.annotate(label_end, point, xytext=(horizontal, 6), textcoords="offset points",
                        ha="right" if horizontal < 0 else "left", color=INK, fontsize=10,
                        weight="bold", zorder=6,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        rmsd = record["comparisons"][pdb_id]["rmsd"]
        ax.set_title(f"({chr(97 + i)}) {label} · {pdb_id}\nCα RMSD {rmsd:.2f} Å", pad=7)
        scale = AnchoredSizeBar(ax.transData, 10, "10 Å", "lower right",
                               pad=0.15, borderpad=0.2, frameon=False, color=MUTED)
        ax.add_artist(scale)
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0.17)
        ax.set_axis_off()
    handles = [Line2D([], [], color=BLUE, linewidth=2.25, label="Experimental Cα trace"),
               Line2D([], [], color=ORANGE, linestyle="--", linewidth=1.6, label="ESMFold2-Fast")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    save(fig, output, "ch30_fold_overlay.pdf")


def plot_ubiquitin_geometry(record, ref, output):
    fig, axes = plt.subplots(1, 2, figsize=(8.45, 3.5), layout="constrained")
    positions = np.asarray(ref["positions"])
    pred_xyz = np.asarray(record["ca"])[positions - 1]
    ref_xyz = np.asarray(ref["ca"])
    length = len(positions)
    contact_model = contact_map(pred_xyz, CONTACT_DISTANCE, CONTACT_SEPARATION)
    contact_exp = contact_map(ref_xyz, CONTACT_DISTANCE, CONTACT_SEPARATION)
    codes = contact_exp.astype(int) + 2 * contact_model
    ax = axes[0]
    ax.imshow(codes, cmap=ListedColormap(CONTACT_COLORS), vmin=0, vmax=3,
              interpolation="none", origin="lower", extent=(0.5, length + 0.5, 0.5, length + 0.5))
    ax.set(title="(a) Cα contacts", xlabel="Sequence position", ylabel="Sequence position")
    ax.set_xticks([1, 20, 40, 60, 76])
    ax.set_yticks([1, 20, 40, 60, 76])
    handles = [Line2D([], [], marker="s", linestyle="", color=color, label=label, markersize=6)
               for color, label in zip(CONTACT_COLORS[1:], ("Experiment", "Prediction", "Both"))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncols=3, frameon=False, fontsize=9, handletextpad=0.25, columnspacing=0.9)

    ax = axes[1]
    measured = np.asarray(record["comparisons"][ref["pdb_id"]]["per_residue_lddt_ca"])
    confidence = np.asarray(record["plddt_ca"])[positions - 1]
    ax.plot(positions, confidence, color=BLUE, linewidth=1.8, label="Predicted pLDDT")
    ax.plot(positions, 100 * measured, color=GREEN, linestyle="--", linewidth=1.6,
            label="Cα lDDT vs. 1UBQ")
    ax.set(title="(b) Local confidence and agreement", xlabel="Sequence position",
           ylabel="Score (0–100)", xlim=(1, length), ylim=(0, 103))
    ax.set_xticks([1, 20, 40, 60, 76])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2, frameon=False,
              fontsize=9, columnspacing=1, handletextpad=0.6)
    save(fig, output, "ch30_ubiquitin_geometry.pdf")


def plot_calmodulin(record, refs, output):
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.30), layout="constrained")
    scores = record["comparisons"]
    for i, pdb_id in enumerate(("1CLL", "1CFD")):
        ref = refs[pdb_id]
        ax = axes[i]
        reference, prediction = projected_traces(record, ref, 0.20, 0.06)
        draw_trace(ax, reference, BLUE, linewidth=2)
        draw_trace(ax, prediction, ORANGE, linestyle="--", linewidth=1.45, zorder=4)
        rmsd = scores[pdb_id]["rmsd"]
        condition = "Calcium bound" if i == 0 else "Calcium free"
        ax.set_title(f"({chr(97 + i)}) {condition}\n{pdb_id} · RMSD {rmsd:.2f} Å", fontsize=10.5)
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0.12)
        ax.set_axis_off()

    length = len(record["pae"])
    ax = axes[2]
    matrix = np.asarray(record["pae"])
    heatmap = ax.imshow(matrix, vmin=0, vmax=32, cmap="magma",
                        interpolation="nearest", origin="lower",
                        extent=(0.5, length + 0.5, 0.5, length + 0.5))
    ax.set(title="(c) Predicted aligned error", xlabel="Sequence position",
           ylabel="Sequence position")
    ax.set_xticks([1, 50, 100, 148])
    ax.set_yticks([1, 50, 100, 148])
    ax.tick_params(labelsize=8.5)
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.05, shrink=0.72)
    cbar.set_label("Å")
    cbar.ax.tick_params(labelsize=8.5)
    handles = [Line2D([], [], color=BLUE, linewidth=2, label="Experimental Cα trace"),
               Line2D([], [], color=ORANGE, linestyle="--", linewidth=1.45, label="Same predicted trace")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, fontsize=9)
    save(fig, output, "ch30_calmodulin_states.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=RECORDS / "embedding_runs.json")
    parser.add_argument("--structures", type=Path, default=RECORDS / "structure_runs.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    embedding = json.loads(args.embeddings.read_text(encoding="utf-8"))["result"]
    structures = json.loads(args.structures.read_text(encoding="utf-8"))["result"]
    refs = load_references()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_embeddings(embedding, args.output_dir)
    plot_overlays(structures, refs, args.output_dir)
    plot_ubiquitin_geometry(structures["1UBQ"], refs["1UBQ"], args.output_dir)
    plot_calmodulin(structures["1CLL"], refs, args.output_dir)


if __name__ == "__main__":
    main()
