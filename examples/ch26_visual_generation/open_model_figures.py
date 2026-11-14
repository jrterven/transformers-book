"""Reproduce Appendix F's image figures using public DDPM weights.

No training data, private images, or remote Python code are loaded. The model
revisions below are fixed, and only safetensors weights are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np


MODELS = {
    "google/ddpm-cat-256": "82ca0d5db4a5ec6ff0e9be8d86852490bc18a3d9",
    "google/ddpm-cifar10-32": "267b167dc01f0e4e61923ea244e8b988f84deb80",
}


def checkpoint_level(timestep: int) -> int:
    """Map the checkpoint's zero-based noise indices to the book's t=1..T."""
    return int(timestep) + 1


def forward_chain(clean, betas, seed, capture_levels):
    """Use independent Gaussian increments, rather than shared marginal noise."""
    rng = np.random.default_rng(seed)
    state = np.asarray(clean, dtype=np.float64).copy()
    captures = {0: state.copy()} if 0 in capture_levels else {}
    for level, beta in enumerate(np.asarray(betas, dtype=np.float64), start=1):
        if not 0 < beta < 1:
            raise ValueError("every forward beta must lie strictly inside (0, 1)")
        state = np.sqrt(1 - beta) * state + np.sqrt(beta) * rng.standard_normal(state.shape)
        if level in capture_levels:
            captures[level] = state.copy()
    return captures


def displayed_rgb(sample):
    """Use the same fixed [-1,1] display mapping for every panel."""
    sample = np.asarray(sample)
    if sample.ndim != 3 or sample.shape[0] != 3:
        raise ValueError("expected one RGB sample in channel-first order")
    return np.clip(sample.transpose(1, 2, 0) / 2 + 0.5, 0, 1)


def save_figures(output, forward, reverse, estimates, generated, seeds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    levels = sorted(forward)
    fig, axes = plt.subplots(1, len(levels), figsize=(12, 2.65), layout="constrained")
    for ax, level in zip(axes, levels):
        ax.imshow(displayed_rgb(forward[level]), interpolation="nearest")
        ax.set_title(f"t = {level}", fontsize=16, color="#3976AF", pad=9)
        ax.axis("off")
    fig.savefig(output / "app_f_forward_real.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    levels = sorted(reverse, reverse=True)
    fig, axes = plt.subplots(2, len(levels), figsize=(12, 5.05), layout="constrained")
    for col, level in enumerate(levels):
        for row, values in enumerate((reverse, estimates)):
            axes[row, col].imshow(displayed_rgb(values[level]), interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
        axes[0, col].set_title(f"t = {level}", fontsize=16, color="#3976AF", pad=9)
    axes[0, 0].set_ylabel("Current sample", fontsize=15, labelpad=9)
    axes[1, 0].set_ylabel("Clean estimate", fontsize=15, labelpad=9)
    fig.savefig(output / "app_f_reverse_real.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(generated), figsize=(9, 3.4), layout="constrained")
    for ax, image, seed in zip(np.atleast_1d(axes), generated, seeds):
        ax.imshow(displayed_rgb(image), interpolation="nearest")
        ax.set_title(f"Seed {seed}", fontsize=14, color="#46856E", pad=9)
        ax.axis("off")
    fig.savefig(output / "app_f_generated_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="google/ddpm-cat-256")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261110, 20261108, 20261109])
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if args.steps < 5 or 1000 % args.steps or args.steps % 5:
        parser.error("steps must divide 1000 and be a multiple of five")

    import torch
    from diffusers import DDIMScheduler, UNet2DModel
    from huggingface_hub import snapshot_download

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    revision = MODELS[args.model]
    snapshot = Path(snapshot_download(
        args.model, revision=revision, token=False,
        allow_patterns=["config.json", "scheduler_config.json", "diffusion_pytorch_model.safetensors", "README.md"],
    ))
    print(f"Loading {args.model}@{revision} on {device}", flush=True)
    model = UNet2DModel.from_pretrained(snapshot, use_safetensors=True).to(device=device, dtype=torch.float32).eval()
    scheduler = DDIMScheduler.from_pretrained(snapshot, timestep_spacing="trailing", clip_sample=True)
    size = model.config.sample_size
    height, width = (size, size) if isinstance(size, int) else tuple(size)
    reverse, estimates, generated = {}, {}, []
    raw = {}
    capture_indices = set(range(0, args.steps, args.steps // 5))
    started = time.monotonic()
    with torch.inference_mode():
        for seed_index, seed in enumerate(args.seeds):
            scheduler.set_timesteps(args.steps, device=device)
            # CPU RNG keeps the initial noise draw independent of device RNG implementations.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            sample = torch.randn((1, 3, height, width), generator=generator, dtype=torch.float32).to(device)
            for index, timestep in enumerate(scheduler.timesteps):
                epsilon = model(sample, timestep).sample
                result = scheduler.step(epsilon, timestep, sample, eta=0, use_clipped_model_output=True)
                if seed_index == 0 and index in capture_indices:
                    level = checkpoint_level(timestep.item())
                    reverse[level] = sample[0].cpu().numpy().copy()
                    estimates[level] = result.pred_original_sample[0].cpu().numpy().copy()
                sample = result.prev_sample
                if (index + 1) % 20 == 0:
                    print(f"seed {seed}: {index + 1}/{args.steps} denoiser calls", flush=True)
            final = sample[0].cpu().numpy().copy()
            if not np.isfinite(final).all():
                raise FloatingPointError("nonfinite generated sample")
            generated.append(final)
            if seed_index == 0:
                reverse[0] = final.copy()
                estimates[0] = final.copy()

    forward = forward_chain(np.clip(generated[0], -1, 1), scheduler.betas.cpu().numpy(),
                            seed=20261111, capture_levels={0, 50, 100, 200, 400, 1000})
    for group, values in (("forward", forward), ("reverse", reverse), ("estimate", estimates)):
        raw.update({f"{group}_{level}": array for level, array in values.items()})
    raw.update({f"generated_{seed}": image for seed, image in zip(args.seeds, generated)})
    np.savez_compressed(args.output / "diffusion_arrays.npz", **raw)
    save_figures(args.output, forward, reverse, estimates, generated, args.seeds)
    packages = ["torch", "diffusers", "huggingface_hub", "safetensors", "numpy", "Pillow", "matplotlib"]
    manifest = {
        "model": args.model, "revision": revision, "model_card_license": "apache-2.0",
        "weights_sha256": hashlib.sha256((snapshot / "diffusion_pytorch_model.safetensors").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": device, "dtype": "float32", "seeds": args.seeds, "forward_seed": 20261111,
        "image_shape_chw": [3, height, width], "sampler": "DDIM", "eta": 0,
        "denoiser_calls_per_image": args.steps, "clip_clean_estimate": True,
        "use_clipped_model_output": True, "timestep_spacing": "trailing",
        "model_timesteps": [int(t) for t in scheduler.timesteps.cpu().tolist()],
        "displayed_levels": sorted(reverse, reverse=True),
        "forward_displayed_levels": sorted(forward), "forward_increments": "independent",
        "display_mapping": "clip(x/2+0.5,0,1); fixed for every panel",
        "scheduler_config": dict(scheduler.config),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "elapsed_seconds": time.monotonic() - started,
        "figures_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob("app_f_*.png"))},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": manifest["elapsed_seconds"],
                      "figure_files": list(manifest["figures_sha256"])}), flush=True)


if __name__ == "__main__":
    main()
