"""Measure DINOv3 patch descriptors and V-JEPA 2 action scores on CPU."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

from huggingface_hub import snapshot_download
import numpy as np
from safetensors.torch import load_file
import timm
import torch
from torch.nn import functional as F
from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

from .demo_media import ASSETS, WINDOWS, asset, image_pair, sampled_frames, verify_sha256


DINO = {
    "repo": "timm/vit_small_patch16_dinov3.lvd1689m",
    "revision": "3bf4720a82ec2066db88137180ff1f83a675cef0",
    "sha256": "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040",
}
VJEPA = {
    "repo": "facebook/vjepa2-vitl-fpc16-256-ssv2",
    "revision": "4aa02df83918538fc21cfaf576382fa20e489a80",
    "sha256": "35aeab96e8513d9cc60ee08c1642dd121c86452c926ce3599b3317896a320396",
}
IMAGE_SIZE = 256
IMAGE_PATCH = 16
SHIFT = 32
FILL = (124, 116, 104)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
QUERY_RC = (3, 7)
VIDEO_LABEL_IDS = {"folding": 14, "unfolding": 172}


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def checkpoint(spec, offline):
    path = Path(snapshot_download(
        spec["repo"], revision=spec["revision"],
        allow_patterns=["*.json", "model.safetensors"], local_files_only=offline,
    ))
    verify_sha256(path / "model.safetensors", spec["sha256"])
    return path


def parameters(model):
    return sum(p.numel() for p in model.parameters())


def argmax_rc(scores):
    return [int(index) for index in np.unravel_index(scores.argmax(), scores.shape)]


@torch.inference_mode()
def observe_image(path, snapshot):
    model = timm.create_model(
        "vit_small_patch16_dinov3", pretrained=False, num_classes=0, img_size=IMAGE_SIZE,
    ).eval()
    model.load_state_dict(load_file(snapshot / "model.safetensors"), strict=True)
    original, shifted = image_pair(path, IMAGE_SIZE, SHIFT, FILL)
    rgb = torch.from_numpy(np.stack([original, shifted])).permute(0, 3, 1, 2).float() / 255
    mean = rgb.new_tensor(IMAGE_MEAN)[None, :, None, None]
    std = rgb.new_tensor(IMAGE_STD)[None, :, None, None]
    pixels = (rgb - mean) / std
    local = model.patch_embed(pixels)
    features = model.forward_features(pixels)
    dense = features[:, model.num_prefix_tokens:]
    batch, rows, cols, width = local.shape
    dense = dense.reshape(batch, rows, cols, width)

    r, c = QUERY_RC
    col_shift = SHIFT // IMAGE_PATCH
    target = (r, c + col_shift)
    kernel = model.patch_embed.proj
    patch = pixels[0, :, r * IMAGE_PATCH:(r + 1) * IMAGE_PATCH, c * IMAGE_PATCH:(c + 1) * IMAGE_PATCH]
    channel_sums = (patch * kernel.weight[0]).sum(dim=(1, 2))
    projection = channel_sums.sum() + kernel.bias[0]
    torch.testing.assert_close(projection, local[0, r, c, 0], atol=2e-6, rtol=2e-6)
    local_differences = (local[0, :, :-col_shift] - local[1, :, col_shift:]).norm(dim=-1)
    dense_differences = (dense[0, :, :-col_shift] - dense[1, :, col_shift:]).norm(dim=-1)
    torch.testing.assert_close(local_differences, torch.zeros_like(local_differences), atol=2e-6, rtol=0)

    unit = F.normalize(dense, dim=-1)
    maps = torch.einsum("brcd,d->brc", unit, unit[0, r, c]).cpu().numpy()
    dots = unit[0, r, c] * unit[1, target[0], target[1]]
    before_after = {
        "input_rgb_equal": bool(np.array_equal(
            original[r * IMAGE_PATCH:(r + 1) * IMAGE_PATCH, c * IMAGE_PATCH:(c + 1) * IMAGE_PATCH],
            shifted[r * IMAGE_PATCH:(r + 1) * IMAGE_PATCH, target[1] * IMAGE_PATCH:(target[1] + 1) * IMAGE_PATCH],
        )),
        "local_max_l2": local_differences.max().item(),
        "context_l2_at_target": dense_differences[r, c].item(),
        "context_median_l2": dense_differences.median().item(),
        "context_max_l2": dense_differences.max().item(),
        "target_cosine": float(maps[1, target[0], target[1]]),
        "original_coordinate_cosine_in_shifted": float(maps[1, r, c]),
        "target_first_four_cosine_products": dots[:4].tolist(),
        "target_remaining_products_sum": dots[4:].sum().item(),
    }
    return {
        "model": {**DINO, "parameters": parameters(model)},
        "preprocess": {"size": IMAGE_SIZE, "mean": IMAGE_MEAN, "std": IMAGE_STD,
                       "shift_pixels": SHIFT, "fill_rgb": FILL, "interpolation": "PIL bicubic"},
        "patch_size": IMAGE_PATCH, "prefix_tokens": model.num_prefix_tokens,
        "image_tensor_shape": list(pixels.shape), "local_shape": list(local.shape),
        "encoder_shape": list(features.shape), "query_rc": list(QUERY_RC), "target_rc": list(target),
        "rgb_sha256": [array_hash(original), array_hash(shifted)],
        "first_red_pixel": {"uint8": int(original[r * IMAGE_PATCH, c * IMAGE_PATCH, 0]),
                            "normalized": patch[0, 0, 0].item(),
                            "kernel_weight": kernel.weight[0, 0, 0, 0].item(),
                            "product": (patch[0, 0, 0] * kernel.weight[0, 0, 0, 0]).item()},
        "projection_channel_sums": channel_sums.tolist(), "projection_bias": kernel.bias[0].item(),
        "projection_coordinate": local[0, r, c, 0].item(),
        "translation": before_after, "original_cosine": maps[0].tolist(),
        "shifted_cosine": maps[1].tolist(), "shifted_argmax_rc": argmax_rc(maps[1]),
    }


def score_record(logits, id2label):
    shifted = logits.double() - logits.max()
    weights = shifted.exp()
    denom = weights.sum()
    p = weights / denom
    ranking = torch.argsort(logits, descending=True, stable=True).tolist()
    return {
        "logits": logits.tolist(), "max_logit": logits.max().item(),
        "shifted_exp_sum": denom.item(),
        "top5": [{"id": idx, "label": id2label[idx], "softmax": p[idx].item()}
                 for idx in ranking[:5]],
        "direction": {name: {"id": idx, "rank": ranking.index(idx) + 1,
                             "logit": logits[idx].item(), "softmax": p[idx].item()}
                      for name, idx in VIDEO_LABEL_IDS.items()},
        "unfolding_minus_folding": (logits[VIDEO_LABEL_IDS["unfolding"]] - logits[VIDEO_LABEL_IDS["folding"]]).item(),
    }


def pair_metrics(clips, method):
    correct = paired = full_top1 = 0
    for clip in clips:
        hits = []
        for direction in ("original", "reversed"):
            row = clip[direction]
            scores = row[method]
            prediction = max(scores["direction"], key=lambda name: scores["direction"][name]["logit"])
            hits.append(prediction == row["expected"])
            full_top1 += scores["top5"][0]["id"] == VIDEO_LABEL_IDS[row["expected"]]
        correct += sum(hits)
        paired += all(hits)
    return {"correct": correct, "views": 2 * len(clips), "both_correct": paired,
            "pairs": len(clips), "full_vocabulary_top1_correct": full_top1}


@torch.inference_mode()
def observe_video(path, snapshot):
    model, load_info = VJEPA2ForVideoClassification.from_pretrained(
        snapshot, use_safetensors=True, trust_remote_code=False, local_files_only=True,
        attn_implementation="sdpa", dtype=torch.float32, output_loading_info=True,
    )
    if any(load_info.values()):
        raise RuntimeError(f"Unexpected checkpoint loading report: {load_info}")
    model.eval()
    processor = AutoVideoProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    config = model.config
    id2label = config.id2label
    for name, idx in VIDEO_LABEL_IDS.items():
        if id2label[idx] != f"{name.title()} [something]":
            raise ValueError(f"Unexpected label at index {idx}: {id2label[idx]}")
    clip_indices = [list(range(start, start + config.frames_per_clip * step, step))
                    for _, start, step, _ in WINDOWS]
    indices_all = sorted({index for indices in clip_indices for index in indices})
    frames_all, times_all = sampled_frames(path, indices_all)
    frame_offsets = {index: offset for offset, index in enumerate(indices_all)}
    clips = []
    for (name, _, _, label), indices in zip(WINDOWS, clip_indices, strict=True):
        offsets = [frame_offsets[index] for index in indices]
        frames = frames_all[offsets]
        timestamps = [times_all[offset] for offset in offsets]
        rgb = torch.from_numpy(frames).permute(0, 3, 1, 2)
        pixel_values = processor(rgb, do_sample_frames=False, return_tensors="pt").pixel_values_videos
        # Each image is scored by the same video checkpoint with that frame repeated twice.
        per_frame = pixel_values[0].unsqueeze(1).expand(-1, config.tubelet_size, -1, -1, -1).contiguous()
        still_logits = model(pixel_values_videos=per_frame).logits.double()
        still_forward = still_logits.mean(dim=0)
        still_backward = still_logits.flip(0).mean(dim=0)
        torch.testing.assert_close(still_forward, still_backward, rtol=0, atol=1e-12)
        encoder = model.vjepa2.encoder
        token_shape = list(encoder.embeddings(pixel_values).shape)
        reversed_label = "folding" if label == "unfolding" else "unfolding"
        clip = {
            "name": name, "indices": indices, "timestamps_s": timestamps,
            "decoded_rgb_shape": list(frames.shape), "model_input_shape": list(pixel_values.shape),
            "tubelet_embedding_shape": token_shape,
            "still_model_input_shape": list(per_frame.shape),
            "still_direction_logits": {name: still_logits[:, idx].tolist() for name, idx in VIDEO_LABEL_IDS.items()},
            "rgb_frame_sha256": [array_hash(frame) for frame in frames],
        }
        for direction, x, expected, baseline in (
            ("original", pixel_values, label, still_forward),
            ("reversed", pixel_values.flip(1), reversed_label, still_backward),
        ):
            scores = model(pixel_values_videos=x).logits[0]
            clip[direction] = {"expected": expected, "temporal": score_record(scores, id2label),
                               "frame_average": score_record(baseline, id2label)}
        clips.append(clip)
    return {
        "model": {**VJEPA, "parameters": parameters(model)},
        "preprocess": {"resize_short_edge": processor.size.shortest_edge,
                       "crop": [processor.crop_size.height, processor.crop_size.width],
                       "resample": int(processor.resample),
                       "mean": processor.image_mean, "std": processor.image_std},
        "tubelet_size": config.tubelet_size, "patch_size": config.patch_size,
        "heads": config.num_attention_heads, "width": config.hidden_size,
        "label_names": id2label, "clips": clips,
        "pair_metrics": {method: pair_metrics(clips, method) for method in ("temporal", "frame_average")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/ch21-pretrained.json"))
    parser.add_argument("--media-dir", type=Path, default=Path("output/ch21-media"))
    parser.add_argument("--offline", action="store_true", help="Require cached weights and media")
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    image = observe_image(asset(args.media_dir, "astronaut.png", offline=args.offline), checkpoint(DINO, args.offline))
    print("Measured image projection and descriptor maps.", flush=True)
    video = observe_video(asset(args.media_dir, "fan.webm", offline=args.offline), checkpoint(VJEPA, args.offline))
    record = {
        "settings": {"device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
                     "seed": 21, "attention": "sdpa", "ffmpeg": subprocess.check_output(
                         ["ffmpeg", "-version"], text=True).splitlines()[0],
                     "packages": {p: version(p) for p in ("torch", "torchvision", "transformers", "timm", "pillow", "numpy")}},
        "assets": ASSETS, "image": image, "video": video,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(video["pair_metrics"], indent=2), flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
