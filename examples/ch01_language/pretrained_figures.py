"""Observe a frozen DistilGPT2 checkpoint for Chapter 1's three figures.

The hand-calculated NumPy model remains independent of this optional experiment.
Only built-in Transformers code and safetensors weights are used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from examples.ch01_language.model import stable_softmax


MODEL = "distilbert/distilgpt2"
REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
TOKEN_TEXTS = ("the red car", "The red car.", "unhelpfulness", "red", " red")
CONTEXT_TEXTS = ("the red car", "the blue car", "the red bike")
PROMPTS = ("She drove the red", "She wore the red")
CANDIDATES = (" car", " dress", " truck", " shirt")
COLORS = ("#267C98", "#AF5482", "#E49A3A", "#6B8C52")


def candidate_distribution(logits, candidate_ids):
    """Return full-vocabulary probabilities for selected IDs and remaining mass."""
    z = np.asarray(logits, dtype=np.float64)
    ids = np.asarray(candidate_ids)
    if z.ndim != 1 or not z.size:
        raise ValueError("expected one nonempty vocabulary row")
    if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
        raise ValueError("candidate IDs must be a nonempty integer vector")
    if np.any(ids < 0) or np.any(ids >= z.size) or np.unique(ids).size != ids.size:
        raise ValueError("candidate IDs must be distinct vocabulary entries")
    probabilities = stable_softmax(z)
    remaining = np.ones(z.size, dtype=bool)
    remaining[ids] = False
    return probabilities[ids], float(probabilities[remaining].sum())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readable(tokenizer, token_id):
    return tokenizer.decode([int(token_id)]).replace(" ", "·")


def observe(cache_dir, offline):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, GPT2LMHeadModel

    torch.set_num_threads(4)
    common = dict(revision=REVISION, cache_dir=str(cache_dir),
                  local_files_only=offline, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, **common)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL, **common, use_safetensors=True, attn_implementation="eager"
    ).eval().float().requires_grad_(False)
    assert model.config._commit_hash == REVISION
    weight_file = hf_hub_download(MODEL, "model.safetensors", revision=REVISION,
                                 cache_dir=str(cache_dir), local_files_only=True)
    token_rows = []
    for text in TOKEN_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.decode(ids) == text
        token_rows.append(dict(text=text, ids=ids,
                               pieces=[readable(tokenizer, i) for i in ids]))

    inputs, embeddings, hidden = [], [], []
    with torch.inference_mode():
        for text in CONTEXT_TEXTS:
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            ids = encoded["input_ids"]
            inputs.append(ids[0].numpy())
            embeddings.append(model.get_input_embeddings()(ids)[0].numpy())
            hidden.append(model(**encoded, output_hidden_states=True,
                                use_cache=False).hidden_states[-1][0].numpy())
        logits = []
        prompt_ids = []
        for text in PROMPTS:
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            prompt_ids.append(encoded["input_ids"][0].numpy())
            logits.append(model(**encoded, use_cache=False).logits[0, -1].numpy())

    assert inputs[0][-1] == inputs[1][-1]
    np.testing.assert_array_equal(embeddings[0][-1], embeddings[1][-1])
    assert not np.allclose(hidden[0][-1], hidden[1][-1])
    np.testing.assert_array_equal(inputs[0][:2], inputs[2][:2])
    np.testing.assert_allclose(hidden[0][1], hidden[2][1], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(prompt_ids[0][-2:], prompt_ids[1][-2:])
    candidate_ids = [tokenizer.encode(s, add_special_tokens=False) for s in CANDIDATES]
    assert all(len(ids) == 1 for ids in candidate_ids)
    candidate_ids = [ids[0] for ids in candidate_ids]
    probabilities = stable_softmax(np.asarray(logits))
    np.testing.assert_allclose(probabilities.sum(-1), 1, rtol=0, atol=1e-12)
    grouped = [candidate_distribution(z, candidate_ids) for z in logits]
    np.testing.assert_allclose([p.sum() + r for p, r in grouped], 1, atol=1e-12)
    report = dict(
        model=MODEL, revision=REVISION, weights_sha256=sha256(weight_file),
        mode="eval; frozen weights; CPU float32 forward pass; no sampling",
        normalization="float64 softmax of the observed float32 logits, full vocabulary",
        vocabulary_size=len(tokenizer), embedding_width=model.config.n_embd,
        tokenization=token_rows,
        context_texts=list(CONTEXT_TEXTS), context_ids=[a.tolist() for a in inputs],
        displayed_coordinates=list(range(1, 13)),
        static_embedding_max_difference=float(np.max(np.abs(embeddings[0][-1] - embeddings[1][-1]))),
        contextual_hidden_max_difference=float(np.max(np.abs(hidden[0][-1] - hidden[1][-1]))),
        earlier_hidden_future_change_max_difference=float(np.max(np.abs(hidden[0][1] - hidden[2][1]))),
        candidates=[dict(text=s, id=i) for s, i in zip(CANDIDATES, candidate_ids)],
        predictions=[dict(prompt=s, ids=ids.tolist(), candidate_probabilities=p.tolist(),
                          remaining_mass=r,
                          top_five=[dict(text=readable(tokenizer, i), id=int(i),
                                         probability=float(full[i]))
                                    for i in np.argsort(full)[-5:][::-1]])
                     for s, ids, (p, r), full in zip(PROMPTS, prompt_ids, grouped, probabilities)],
        software={name: importlib.metadata.version(name) for name in
                  ("torch", "transformers", "tokenizers", "huggingface_hub",
                   "safetensors", "numpy", "matplotlib", "Pillow")},
    )
    arrays = dict(lookup_vectors=np.stack([e[-1] for e in embeddings]),
                  final_hidden_vectors=np.stack([h[-1] for h in hidden]),
                  earlier_hidden_control=np.stack([hidden[0][1], hidden[2][1]]),
                  logits=np.stack(logits), probabilities=probabilities)
    return report, arrays


def draw_figures(report, arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    paths = []

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ax.set(xlim=(0, 10.2), ylim=(-0.15, 5.7))
    ax.axis("off")
    ax.text(0, 5.4, "Actual text", weight="bold", color="#27374B")
    ax.text(3.5, 5.4, "Decoded token pieces and vocabulary IDs", weight="bold", color="#27374B")
    for row, record in enumerate(report["tokenization"]):
        y = 4.55 - row
        ax.text(0, y + .05, repr(record["text"]), va="center", family="monospace", fontsize=11)
        x = 3.5
        for piece, token_id in zip(record["pieces"], record["ids"]):
            width = max(.95, .21 * len(piece) + .35)
            ax.add_patch(FancyBboxPatch((x, y - .34), width, .73,
                         boxstyle="round,pad=0.04", facecolor="#EDF4F7", edgecolor="#7A9BAA"))
            ax.text(x + width / 2, y + .13, piece, ha="center", va="center", family="monospace", fontsize=11)
            ax.text(x + width / 2, y - .15, str(token_id), ha="center", va="center", fontsize=9, color="#526576")
            x += width + .16
    ax.text(0, -.04, "· marks an actual space in the decoded token. Quotes delimit the input text.", color="#526576", fontsize=9)
    fig.tight_layout()
    path = output / "ch01_pretrained_tokens.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.3), constrained_layout=True)
    for ax, values, title in zip(axes,
            (arrays["lookup_vectors"][:2, :12], arrays["final_hidden_vectors"][:2, :12]),
            ("(a) Token lookup: same ' car' ID, same vector",
             "(b) Final hidden state at ' car': earlier context changes the vector")):
        limit = float(np.max(np.abs(values)))
        im = ax.imshow(values, cmap="PuOr_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_yticks((0, 1), CONTEXT_TEXTS[:2])
        ax.set_xticks(range(12), range(1, 13))
        ax.set_title(title, loc="left", fontsize=11, pad=9)
        for (r, c), value in np.ndenumerate(values):
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(value) > .65 * limit else "#202020")
        fig.colorbar(im, ax=ax, shrink=.9, label="Coordinate value")
    axes[-1].set_xlabel("First 12 of 768 coordinates (numbered from 1); each panel has its own color scale")
    path = output / "ch01_pretrained_context.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig = plt.figure(figsize=(8.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=(3, 1.6))
    records = report["predictions"]
    upper = max(max(r["candidate_probabilities"]) for r in records) * 1.25
    for col, record in enumerate(records):
        ax = fig.add_subplot(gs[0, col])
        p = np.array(record["candidate_probabilities"])
        ax.barh(range(4), p, color=COLORS, height=.6)
        ax.set_yticks(range(4), [s.strip() for s in CANDIDATES])
        ax.invert_yaxis()
        ax.set(xlim=(0, upper), xlabel="Probability in the full vocabulary")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(repr(record["prompt"]), fontsize=11, loc="left")
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        for y, value in enumerate(p):
            ax.text(value + upper * .02, y, f"{100 * value:.2f}%", va="center", fontsize=9)
    ax = fig.add_subplot(gs[1, :])
    for row, record in enumerate(records):
        left = 0
        for value, color in zip(record["candidate_probabilities"], COLORS):
            ax.barh(row, value, left=left, color=color, height=.55)
            left += value
        residual = record["remaining_mass"]
        ax.barh(row, residual, left=left, color="#CBD0D5", height=.55)
        ax.text(left + residual / 2, row, f"All other {report['vocabulary_size'] - 4:,} tokens: {100 * residual:.2f}%",
                ha="center", va="center", fontsize=9)
    ax.set_yticks((0, 1), ("drove", "wore"))
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="All vocabulary entries together = 100%")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    path = output / "ch01_pretrained_predictions.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Hugging Face hub cache directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, arrays = observe(args.cache_dir, args.offline)
    array_path = args.output_dir / "ch01_pretrained_observations.npz"
    np.savez_compressed(array_path, **arrays)
    figures = draw_figures(report, arrays, args.output_dir)
    report["source_sha256"] = sha256(__file__)
    report["artifacts"] = {p.name: sha256(p) for p in [array_path, *figures]}
    (args.output_dir / "PRETRAINED_FIGURES_RUN.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
