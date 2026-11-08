"""Make measured attention, loss-gradient, and convolution figures.

Only built-in model implementations are used. No remote Python is loaded.
Weights stay fixed: these are observations, not model training or benchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import urllib.request

import numpy as np


LM_NAME = "distilbert/distilgpt2"
LM_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
VISION_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
VISION_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
ATTENTION_TEXT = "The small cat sat on the warm mat."
LOSS_PROMPT = "The cat sat on the"
LOSS_TARGET = " mat"
HEADS = (0, 1)  # Zero-based; the figure numbers heads from one.
CHANNELS = (1, 14, 61)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_probabilities(logits: np.ndarray) -> np.ndarray:
    """Normalize one finite, nonempty row without overflowing exponentials."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or not z.size or not np.isfinite(z).all():
        raise ValueError("Expected a finite, nonempty logit row")
    e = np.exp(z - z.max())
    return e / e.sum()


def cache_payload_bytes(layers: int, batch: int, heads: int, tokens: int,
                        key_width: int, value_width: int, itemsize: int) -> int:
    """Logical K/V payload; excludes backing-storage capacity and other arrays."""
    sizes = (layers, batch, heads, tokens, key_width, value_width, itemsize)
    if any(int(x) != x or x <= 0 for x in sizes):
        raise ValueError("Dimensions and item size must be positive integers")
    return layers * batch * heads * tokens * (key_width + value_width) * itemsize


def cache_pairs(cache):
    """Read K/V tensors from the documented legacy representation."""
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return tuple((layer[0], layer[1]) for layer in cache)


def readable_token(tokenizer, token_id: int) -> str:
    # A middle dot represents a leading space; punctuation is kept separately.
    return tokenizer.decode([int(token_id)]).replace(" ", "·")


def inspect_language_model(cache_dir: Path, offline: bool):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, GPT2LMHeadModel

    common = dict(revision=LM_REVISION, cache_dir=str(cache_dir),
                  local_files_only=offline, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(LM_NAME, **common)
    model = GPT2LMHeadModel.from_pretrained(
        LM_NAME, **common, use_safetensors=True, attn_implementation="eager"
    ).eval().float()
    assert model.config._commit_hash == LM_REVISION
    weight_file = Path(hf_hub_download(LM_NAME, "model.safetensors", revision=LM_REVISION,
                                     cache_dir=str(cache_dir), local_files_only=True))
    model.requires_grad_(False)
    encoded = tokenizer(ATTENTION_TEXT, return_tensors="pt")
    with torch.inference_mode():
        result = model(**encoded, output_attentions=True, use_cache=True)
    attentions = np.stack([a[0].cpu().numpy() for a in result.attentions])
    np.testing.assert_allclose(attentions.sum(-1), 1, rtol=0, atol=4e-7)
    assert np.max(np.abs(np.triu(attentions, k=1))) == 0
    cache_records = []
    for repeats in (1, 2):
        ids = encoded["input_ids"].repeat(1, repeats)
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=True)
        pairs = cache_pairs(out.past_key_values)
        measured = sum(t.numel() * t.element_size() for pair in pairs for t in pair)
        k, v = pairs[0]
        expected = cache_payload_bytes(len(pairs), k.shape[0], k.shape[1],
                                       k.shape[2], k.shape[3], v.shape[3],
                                       k.element_size())
        assert measured == expected
        cache_records.append(dict(tokens=int(ids.shape[1]), layers=len(pairs),
                                  key_shape=list(k.shape), value_shape=list(v.shape),
                                  dtype=str(k.dtype), bytes=measured,
                                  kib=measured / 1024))

    prompt = tokenizer(LOSS_PROMPT, return_tensors="pt")
    target_ids = tokenizer.encode(LOSS_TARGET, add_special_tokens=False)
    assert len(target_ids) == 1
    target = target_ids[0]
    with torch.inference_mode():
        raw_logits = model(**prompt, use_cache=False).logits[0, -1].cpu().numpy()
    # Differentiate the final loss operation in FP64, using the observed logits.
    # No parameter gradients or parameter updates are performed.
    z = torch.tensor(raw_logits, dtype=torch.float64, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(z[None], torch.tensor([target]))
    loss.backward()
    gradient = z.grad.numpy()
    probabilities = stable_probabilities(raw_logits)
    expected_gradient = probabilities.copy()
    expected_gradient[target] -= 1
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=3e-15)
    assert abs(gradient.sum()) < 1e-12
    top = np.argsort(probabilities)[::-1][:5].tolist()
    selected = top if target in top else top + [target]
    # The offset is applied in FP32, so input rounding can change some logits.
    shifted = raw_logits + np.float32(1000)
    stable_shifted = stable_probabilities(shifted)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(shifted)
        naive = e / e.sum()
    metadata = dict(
        model=LM_NAME, revision=LM_REVISION, weights_sha256=sha256(weight_file),
        dtype="float32", device="cpu",
        eval_mode=True, attention_implementation="eager", weights_updated=False,
        text=ATTENTION_TEXT, token_ids=encoded["input_ids"][0].tolist(),
        tokens=[readable_token(tokenizer, t) for t in encoded["input_ids"][0]],
        attention_shape=list(attentions.shape), displayed_layer=6,
        displayed_heads=[h + 1 for h in HEADS],
        maximum_row_sum_error=float(np.abs(attentions.sum(-1) - 1).max()),
        maximum_future_weight=0.0, cache=cache_records,
        loss_prompt=LOSS_PROMPT, supplied_target=LOSS_TARGET, target_id=target,
        vocabulary_size=len(probabilities), loss=float(loss.detach()),
        selected_tokens=[readable_token(tokenizer, t) for t in selected],
        selected_ids=selected,
        selected_probabilities=probabilities[selected].tolist(),
        selected_gradients=gradient[selected].tolist(),
        all_gradient_sum=float(gradient.sum()),
        precision_check=dict(offset=1000, offset_dtype="float32",
                             naive_finite_entries=int(np.isfinite(naive).sum()),
                             stable_sum=float(stable_shifted.sum()),
                             max_probability_change=float(np.abs(
                                 stable_shifted - probabilities).max()))
    )
    arrays = dict(attention=attentions, logits=raw_logits, probabilities=probabilities,
                  logit_gradient=gradient, selected_ids=np.array(selected))
    return metadata, arrays


def inspect_convolution(reference: Path, cache_dir: Path, offline: bool):
    import torch
    from PIL import Image

    weight_file = cache_dir / "resnet18-f37072fd.pth"
    if not weight_file.exists():
        if offline:
            raise FileNotFoundError(weight_file)
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(VISION_URL, weight_file)
    if sha256(weight_file) != VISION_SHA256:
        raise ValueError("ResNet weights failed the pinned SHA-256 check")
    weights = torch.load(weight_file, weights_only=True, map_location="cpu")["conv1.weight"].detach()
    with Image.open(reference) as source:
        rgb = source.convert("RGB")
        if rgb.size != (256, 256):
            raise ValueError("The supplied synthetic reference must be 256 by 256")
        cropped = rgb.crop((16, 16, 240, 240))
        image = np.asarray(cropped).astype(np.float32) / 255
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    x = torch.from_numpy(((image - mean) / std).transpose(2, 0, 1).copy())[None]
    with torch.inference_mode():
        response = torch.nn.functional.conv2d(x, weights, stride=2, padding=3)
    assert response.shape == (1, 64, 112, 112)
    # Check boundaries and an interior pixel with an explicit patch dot product.
    padded = np.pad(x.numpy()[0], ((0, 0), (3, 3), (3, 3)))
    for ch in CHANNELS:
        for row, col in ((0, 0), (30, 41), (111, 111)):
            patch = padded[:, 2 * row:2 * row + 7, 2 * col:2 * col + 7]
            direct = (patch.astype(np.float64)
                      * weights[ch].numpy().astype(np.float64)).sum()
            np.testing.assert_allclose(response[0, ch, row, col], direct,
                                       rtol=0, atol=2e-6)
    meta = dict(model="ResNet-18", weights="IMAGENET1K_V1", source=VISION_URL,
                weights_sha256=VISION_SHA256, source_image_sha256=sha256(reference),
                input_provenance="Appendix F DDPM output, seed 20261110",
                crop=[16, 16, 240, 240], mean=mean.tolist(), std=std.tolist(),
                input_shape=list(x.shape), weight_shape=list(weights.shape),
                response_shape=list(response.shape), stride=2, zero_padding=3,
                displayed_channels=list(CHANNELS), response_stage="conv1, before batch normalization and ReLU",
                dtype="float32", device="cpu", weights_updated=False,
                shared_display_limit=float(np.abs(response.numpy()[0, list(CHANNELS)]).max()))
    return meta, dict(reference_crop=image, conv_response=response.numpy()[0],
                     conv_selected_weights=weights[list(CHANNELS)].numpy())


def plot_observations(metadata: dict, arrays: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    lm = metadata["language_model"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for ax, head in zip(axes, HEADS):
        im = ax.imshow(arrays["attention"][5, head], vmin=0, vmax=1,
                       cmap="Blues", interpolation="nearest")
        ticks = np.arange(len(lm["tokens"]))
        ax.set_xticks(ticks, lm["tokens"], rotation=55, ha="right")
        ax.set_yticks(ticks, lm["tokens"])
        ax.set_xlabel("Key position"); ax.set_ylabel("Query position")
        ax.set_title(f"Layer 6, head {head + 1}")
    fig.colorbar(im, ax=axes, label="Attention weight", shrink=.8)
    fig.savefig(output / "app_a_pretrained_attention.png", dpi=260)
    plt.close(fig)

    ids = arrays["selected_ids"]
    labels = lm["selected_tokens"]
    colors = ["#d07837" if int(i) == lm["target_id"] else "#3976af" for i in ids]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for ax, values, title, ylabel in zip(axes,
            (arrays["probabilities"][ids], arrays["logit_gradient"][ids]),
            ("Observed next-token probabilities", "Derivative of the supplied-target loss"),
            ("Probability", r"$\partial L / \partial z_j$")):
        ax.bar(np.arange(len(ids)), values, color=colors)
        ax.axhline(0, color="#687580", linewidth=.8)
        ax.set_xticks(np.arange(len(ids)), labels, rotation=30, ha="right")
        ax.set_title(title); ax.set_ylabel(ylabel)
    fig.savefig(output / "app_b_pretrained_gradients.png", dpi=260)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.05), layout="constrained")
    axes[0].imshow(arrays["reference_crop"])
    axes[0].set_title("Synthetic input crop")
    limit = metadata["convolution"]["shared_display_limit"]
    for ax, channel in zip(axes[1:], CHANNELS):
        im = ax.imshow(arrays["conv_response"][channel], cmap="BrBG",
                       vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_title(f"Filter channel {channel}")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[1:], label="Signed response", shrink=.65)
    fig.savefig(output / "app_e_pretrained_convolution.png", dpi=260)
    plt.close(fig)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path,
                        default=Path(__file__).parent / "assets/reference-cat.png")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path.home() / ".cache/transformers-book/appendix-models")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    lm, language_arrays = inspect_language_model(args.cache_dir, args.offline)
    conv, convolution_arrays = inspect_convolution(args.reference_image, args.cache_dir, args.offline)
    metadata = dict(language_model=lm, convolution=conv,
                    packages={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "tokenizers", "numpy", "matplotlib", "Pillow")},
                    source_sha256=sha256(Path(__file__)))
    arrays = {**language_arrays, **convolution_arrays}
    np.savez_compressed(args.output_dir / "observations.npz", **arrays)
    plot_observations(metadata, arrays, args.output_dir)
    metadata["outputs"] = {p.name: sha256(p) for p in sorted(args.output_dir.glob("*.png"))}
    metadata["arrays_sha256"] = sha256(args.output_dir / "observations.npz")
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
