"""Decode exported model latents using the pinned MLX video VAE implementation."""

import importlib.util
from pathlib import Path

from .generate_images import digest


DECODE_REVISION = "42514d9d47ef6c608889dbc48e6089e607a93cf1"
DECODER_FILES = {
    "wan": ("vae22.py", "94c636e07d16a8977c5d4946d4fa0320f609bd380e4fff9aeb81651f2c978fa1"),
    "lingbot": ("vae.py", "83c355a2a1d8829b1e020c1e7389782f9d15e08b259bab3323496d371e718cee"),
}


def decode(family, decoder_dir, weights_path, latent_path, output, frame_shape):
    import mlx.core as mx
    import numpy as np
    from PIL import Image

    name, expected_hash = DECODER_FILES[family]
    path = decoder_dir / "mlx_video/models/wan_2" / name
    if digest(path) != expected_hash:
        raise ValueError(f"Decoder source must match revision {DECODE_REVISION}")
    spec = importlib.util.spec_from_file_location(f"chapter25_{family}_vae", path)
    vae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vae)

    frames, height, width = frame_shape
    stride, channels = (16, 48) if family == "wan" else (8, 16)
    expected_shape = (width // stride, height // stride, 1 + (frames - 1) // 4, channels, 1)
    with latent_path.open("rb") as stream:
        shape = tuple(np.fromfile(stream, dtype="<i8", count=5))
        if shape != expected_shape:
            raise ValueError(f"Expected latent shape {expected_shape}, got {shape}")
        raw = np.fromfile(stream, dtype="<f4").reshape(shape[::-1])
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite latent values")
    weights = mx.load(str(weights_path))
    if family == "wan":
        decoder = vae.Wan22VAEDecoder()
        decoder.load_weights(list(vae.sanitize_wan22_vae_weights(weights).items()))
        decoder.set_dtype(mx.float16)
        latent = mx.array(raw.transpose(0, 2, 3, 4, 1))
        video = decoder(vae.denormalize_latents(latent).astype(mx.float16))[0]
    else:
        decoder = vae.WanVAE()
        decoder_weights = {name: getattr(decoder, name) for name in ("mean", "std", "inv_std")}
        for key, value in weights.items():
            if key.startswith(("decoder.", "conv2.")):
                if key.endswith(".weight") and value.ndim >= 4:
                    value = value.transpose((0, *range(2, value.ndim), 1))
                decoder_weights[key] = value
        decoder.load_weights(list(decoder_weights.items()))
        decoder.set_dtype(mx.float16)
        # The all-at-once Wan2.1 port emits four frames per latent site.
        # Drop the first three warm-up frames to match Wan's 4n+1 length.
        video = decoder.decode(mx.array(raw).astype(mx.float16))[0, :, 3:].transpose(1, 2, 3, 0)

    array = np.array(video.astype(mx.float32))
    if array.shape != (frames, height, width, 3) or not np.isfinite(array).all():
        raise ValueError(f"Unexpected decoded clip array {array.shape}")
    pixels = np.rint((array + 1) * 127.5).clip(0, 255).astype(np.uint8)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(pixels):
        Image.fromarray(frame).save(output / f"{i:03d}.png")
