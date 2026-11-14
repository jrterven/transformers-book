"""Generate the fixed-prompt FLUX.2 examples and record their latent states."""

import argparse
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
from pathlib import Path


REPOSITORY = "mflux-community/flux2-klein-4b-mflux-q4"
REVISION = "794cd159538149ad9830848508c31f0ea7088e58"
WEIGHT_HASHES = {
    "text_encoder/0.safetensors": "e59ed2b6825cb8ae3d0ba1e8a43575a565d465ce4518b7d79d28020b19b639cd",
    "text_encoder/1.safetensors": "edbdea3187bc4a0ad0e1ecb7152da8e6b344669413d9e45be5e7a2244f398ee3",
    "transformer/0.safetensors": "36796395e3946496f3e118f26fe22594b1d7d871342a59dd7824349d91c08266",
    "transformer/1.safetensors": "9730ae148ca4f5ffc62860748b849a70987f0b7301588248bd98a2af9bc15070",
    "vae/0.safetensors": "688ae0c3d2e93573ce8a6d3561d793cf6de5a4207b2e128cad6cd0d71e43b10f",
}
PROMPT = (
    "A cinematic stop-motion still of a small {color} toy steam locomotive with three black wheels "
    "and a short smokestack on a straight miniature railway. It faces right in a full side view, "
    "centered with ample empty track on both sides. Cream studio backdrop, a pale wooden tabletop, "
    "and a few low paper hills in the distance. Soft afternoon light and crisp handcrafted textures. "
    "Landscape composition, fixed camera, no text."
)
WIDTH, HEIGHT = 640, 384
STEPS, GUIDANCE = 4, 1.0
SCHEDULER = "flow_match_euler_discrete"
CASES = (("red", 2501), ("blue", 2501), ("red", 2502), ("blue", 2502))


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def run(output, offline):
    import mlx.core as mx
    import numpy as np
    from huggingface_hub import snapshot_download
    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
    from mflux.models.flux2.variants import Flux2Klein

    snapshot = Path(snapshot_download(REPOSITORY, revision=REVISION, local_files_only=offline))
    for name, expected in WEIGHT_HASHES.items():
        if digest(snapshot / name) != expected:
            raise ValueError(f"Checkpoint hash differs for {name}")

    class Trace:
        def __init__(self):
            self.states = []
            self.sigmas = []

        def save(self, latents):
            array = np.array(latents.astype(mx.float32))
            self.states.append({
                "shape": list(array.shape),
                "sha256_float32": sha256(array.astype("<f4", copy=False).tobytes()).hexdigest(),
                "rms": float(np.sqrt(np.mean(array.astype(np.float64) ** 2))),
            })

        def call_before_loop(self, seed, prompt, latents, config, **kwargs):
            self.sigmas = config.scheduler.sigmas.tolist()
            self.save(latents)

        def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
            self.save(latents)

    output.mkdir(parents=True, exist_ok=True)
    print(f"Loading {REPOSITORY} at {REVISION}", flush=True)
    model = Flux2Klein(model_path=str(snapshot))
    report = {
        "checkpoint": REPOSITORY, "revision": REVISION, "weight_sha256": WEIGHT_HASHES,
        "versions": {name: version(name) for name in ("mlx", "mflux", "numpy")},
        "width": WIDTH, "height": HEIGHT, "steps": STEPS, "guidance": GUIDANCE,
        "scheduler": SCHEDULER, "cases": [],
    }
    for i, (color, seed) in enumerate(CASES):
        print(f"Generating {color}, seed {seed}", flush=True)
        trace = Trace()
        model.callbacks = CallbackRegistry()
        model.callbacks.register(trace)
        if i == 0:
            steps = StepwiseHandler(model, str(output / "image-states"), Flux2LatentCreator)
            model.callbacks.register(steps)
        prompt = PROMPT.format(color=color)
        result = model.generate_image(seed=seed, prompt=prompt, height=HEIGHT, width=WIDTH,
                                      num_inference_steps=STEPS, guidance=GUIDANCE, scheduler=SCHEDULER)
        path = output / f"train-{color}-{seed}.png"
        result.save(path, export_json_metadata=False)
        report["cases"].append({
            "file": path.name, "sha256": digest(path), "color": color, "seed": seed, "prompt": prompt,
            "sigmas": trace.sigmas, "states": trace.states,
        })
        (output / "image-run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.offline)
