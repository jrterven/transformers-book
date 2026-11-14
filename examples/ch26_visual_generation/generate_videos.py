"""Generate the chapter's video conditions with pinned stable-diffusion.cpp weights."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess

from .decode_videos import DECODE_REVISION, decode
from .generate_images import digest


ENGINE = "b9d7176bf88fe465576e0aafbe47ba253ffcfc2c"
GGML = "d803c238f67915fb825a9aa4df27b99356d51597"
MODEL_FILES = {
    "wan": (
        ("QuantStack/Wan2.2-TI2V-5B-GGUF", "57437632ddd08bdcbd1508c866aa22e126ed51d2",
         "Wan2.2-TI2V-5B-Q4_K_M.gguf", "95b19697b7f98e65b0a543640e9ca7b4dfec32e2a6e3731e8e10708be52655e2"),
        ("QuantStack/Wan2.2-TI2V-5B-GGUF", "57437632ddd08bdcbd1508c866aa22e126ed51d2",
         "VAE/Wan2.2_VAE.safetensors", "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156"),
        ("city96/umt5-xxl-encoder-gguf", "b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7",
         "umt5-xxl-encoder-Q4_K_M.gguf", "17cf97a5bbbc60a646d6105b832b6f657ce904a8a1ad970e4b59df0c67584a40"),
    ),
    "lingbot": (
        ("off-grid-ai/LingBot-Video-Dense-1.3B-GGUF", "5f5be44d9770b1280c22859cbd4efe7ec9fa2a9c",
         "LingBot-Video-Dense-1.3B-Q8_0.gguf", "05036765fb09e68094387b87ba896be96f0cf6e04b646f44197706eb4d01794b"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",
         "split_files/vae/wan_2.1_vae.safetensors", "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),
        ("Qwen/Qwen3-VL-4B-Instruct-GGUF", "1cd86afb9a95c410a6038ab3b40d8b578c892266",
         "Qwen3VL-4B-Instruct-Q4_K_M.gguf", "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a"),
    ),
}
TRAIN = (
    "A handcrafted red toy steam locomotive rolls smoothly from left to right along a straight "
    "miniature track, its black wheels visibly turning. The locomotive starts near the center "
    "and continues toward the right edge. A small wisp of white steam rises from its black chimney. "
    "Locked side-view camera. A pale wooden table, cream backdrop and low folded-paper hills stay still. "
    "Warm soft light, one uninterrupted stop-motion shot."
)
NEGATIVE = "scene cut, camera motion, flicker, warped wheels, duplicate locomotive, text, subtitles, watermark"
LINGBOT = {
    "caption": {
        "comprehensive_description": (
            "A fixed side-view stop-motion shot of a miniature wooden stage. A bright red toy steam locomotive "
            "rolls steadily from left to right along a straight narrow railway on a pale wooden tabletop. "
            "The black wheels rotate and a thin white wisp rises from the smokestack. The small paper hills "
            "in the cream background are stationary. Soft warm studio light, one continuous shot."
        ),
        "camera_info": {
            "color": "Warm", "frame_size": "Medium shot", "shot_type_angle": "Eye level",
            "lens_size": "Medium", "composition": "Full side view", "lighting": "Soft",
            "lighting_type": "Studio daylight",
        },
        "world_knowledge": [],
        "prominent_elements": [{
            "name": "red toy locomotive", "description": "Red locomotive with black wheels and black chimney.",
            "location": "On the horizontal railway in the middle of the shot", "relative_size": "Medium",
            "appearance_details": "Red body, dark wheels, thin white steam", "shape_and_color": "Red and black",
            "material_and_structure": "Painted miniature", "relationship": "Travels on the model railway",
        }],
    },
}


def load_weights(family, offline):
    from huggingface_hub import hf_hub_download

    files = []
    for repo, revision, filename, sha in MODEL_FILES[family]:
        path = Path(hf_hub_download(repo, filename, revision=revision, local_files_only=offline))
        if digest(path) != sha:
            raise ValueError(f"Checkpoint hash differs for {filename}")
        files.append(path)
    return files


def run_case(cli, decoder_dir, case, weights, output):
    diffusion, vae, encoder = weights
    family, name, prompt, source = case
    frame_dir = output / f"{name}-frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    fps, seed = 16, 2503
    if family == "wan":
        frames, steps, width, height, cfg = 49, 22, 640, 384, 5.0
    else:
        frames, steps, width, height, cfg = 25, 14, 320, 192, 3.0
    encoder_flag = "--t5xxl" if family == "wan" else "--llm"
    cmd = [
        str(cli), "-M", "vid_gen", "--backend", "MTL0", "--diffusion-model", str(diffusion),
        "--vae", str(vae), encoder_flag, str(encoder), "-p", prompt, "-n", NEGATIVE,
        "--width", str(width), "--height", str(height), "--video-frames", str(frames), "--fps", str(fps),
        "--steps", str(steps), "--seed", str(seed), "--cfg-scale", str(cfg), "--flow-shift", "3",
        "--sampling-method", "euler", "--flash-attn", "on", "--rng", "cpu",
        "--output", str(frame_dir / "%03d.png"),
    ]
    if source:
        cmd.extend(("--init-img", str(source)))
    latent = output / f"{name}.latent"
    print(f"Generating {name} with {family}", flush=True)
    log_path = output / f"{name}.log"
    # A failed sampling attempt must never fall through to a stale latent.
    latent.unlink(missing_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, env={**os.environ, "SD_CHAPTER25_LATENT": str(latent)},
                                stdout=log, stderr=subprocess.STDOUT)
    # The patch returns before the CLI's frame output stage.
    if not latent.is_file():
        raise RuntimeError(f"No final latent, exit {result.returncode}. See {log_path}")
    decode(family, decoder_dir, vae, latent, frame_dir, (frames, height, width))
    return encode_clip(case, output, latent, frames, steps, width, height, cfg, fps, seed)


def encode_clip(case, output, latent, frames, steps, width, height, cfg, fps, seed):
    family, name, prompt, source = case
    frame_dir = output / f"{name}-frames"
    rendered = [frame_dir / f"{i:03d}.png" for i in range(frames)]
    if not all(path.is_file() for path in rendered):
        raise RuntimeError(f"Expected {frames} consecutively numbered frames in {frame_dir}")
    path = output / f"{name}.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-framerate", str(fps), "-start_number", "0",
        "-i", str(frame_dir / "%03d.png"), "-frames:v", str(frames), "-c:v", "libx264", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], check=True)
    return {
        "family": family, "file": path.name, "sha256": digest(path), "prompt": prompt,
        "negative_prompt": NEGATIVE, "seed": seed, "cfg_scale": cfg, "flow_shift": 3.0,
        "sampling_method": "euler", "rng": "cpu", "flash_attention": "on",
        "steps": steps, "width": width, "height": height, "frames": frames, "fps": fps,
        "source": {"file": source.name, "sha256": digest(source)} if source else None,
        "latent_sha256": digest(latent),
        "frame_sha256": [digest(p) for p in rendered],
        "weights": [{"repository": repo, "revision": rev, "file": f, "sha256": h}
                    for repo, rev, f, h in MODEL_FILES[family]],
    }


def main(cli, decoder_dir, output, image_dir, requested, offline):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg must be on PATH")
    output.mkdir(parents=True, exist_ok=True)
    source = image_dir / "train-red-2501.png"
    cases = {
        "wan-text": ("wan", "wan-text", TRAIN, None),
        "wan-image": ("wan", "wan-image", TRAIN, source),
        "lingbot-text": ("lingbot", "lingbot-text", json.dumps(LINGBOT, ensure_ascii=False), None),
    }
    reports = video_report(cli)
    weights = {}
    for name in requested or cases:
        case = cases[name]
        if case[3] and not case[3].is_file():
            raise FileNotFoundError(case[3])
        family = case[0]
        if family not in weights:
            weights[family] = load_weights(family, offline)
        reports["cases"].append(run_case(cli, decoder_dir, case, weights[family], output))
        (output / "video-run.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


def video_report(cli):
    patch = Path(__file__).with_name("export_video_latent.patch")
    return {
        "engine": ENGINE, "ggml": GGML, "engine_sha256": digest(cli), "export_patch_sha256": digest(patch),
        "backend": "Metal (MTL0)", "decoder_revision": DECODE_REVISION, "decoder_dtype": "float16",
        "mlx": version("mlx"),
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0].split()[2],
        "cases": [],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd-cli", required=True, type=lambda path: Path(path).resolve())
    parser.add_argument("--decoder-source", required=True, type=Path,
                        help="Path to the pinned vga-mlx-video source tree")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--case", action="append", choices=("wan-text", "wan-image", "lingbot-text"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    main(args.sd_cli, args.decoder_source, args.output_dir, args.image_dir, args.case, args.offline)
