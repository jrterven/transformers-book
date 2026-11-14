"""Transcribe and tag the book's PCM fixtures with pinned public checkpoints."""

import argparse
from hashlib import sha1
from importlib.metadata import version
import json
from pathlib import Path
import unicodedata

import numpy as np

from .model import word_error_rate
from .prepare_audio import AUDIO_DIR, CLIPS, SAMPLE_RATE, digest, load_wave


CHECKPOINTS = {
    "parakeet": (
        "nvidia/parakeet-tdt-0.6b-v3",
        "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11",
    ),
    "qwen": (
        "Qwen/Qwen3-ASR-0.6B-hf",
        "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
        "d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35",
    ),
    "melati": (
        "Scicom-intl/Melati-AudioSet-EAT-Large",
        "82762af3cdcfae80bbc964074184c0ffe962f90b",
        "7dd9b4089a84729b353bd6562e1982c3d7ee5cc6c758d7fb419d26dda9c2801e",
    ),
}

# These four small Python files define the released EAT architecture.
EAT_CODE = {
    "configuration_eat.py": "53d1efa86dbda92589228e3f3d48394268312aff",
    "eat_model.py": "4c2bfc54c1638a38f75f873f35c5fbd61c1c04e4",
    "model_core.py": "9eb026f4358203d1640109081c56b64291882db6",
    "modeling_eat.py": "3f49c1b3af170238dcd90ca16c2f6f5e4e47ec9a",
}


def load_checkpoint(name, offline):
    from huggingface_hub import snapshot_download

    repository, revision, weights = CHECKPOINTS[name]
    patterns = ["*.json", "*.jinja", "model.safetensors"]
    if name == "melati":
        patterns.extend(EAT_CODE)
    directory = Path(snapshot_download(repository, revision=revision, allow_patterns=patterns,
                                       local_files_only=offline))
    if digest(directory / "model.safetensors") != weights:
        raise ValueError(f"Weights do not match {repository} at {revision}")
    if name == "melati":
        for filename, expected in EAT_CODE.items():
            data = (directory / filename).read_bytes()
            header = f"blob {len(data)}\0".encode()
            if sha1(header + data).hexdigest() != expected:
                raise ValueError(f"Architecture file does not match the pinned {filename}")
    return directory


def words(text):
    normalized = unicodedata.normalize("NFC", text).casefold()
    return "".join(" " if unicodedata.category(c).startswith("P") else c for c in normalized).split()


def transcribe_parakeet(path, clips, feature_path):
    import torch
    from transformers import AutoModelForTDT, AutoProcessor

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForTDT.from_pretrained(path, use_safetensors=True, local_files_only=True).eval()
    extractor = processor.feature_extractor
    encoder_config = model.config.encoder_config
    results = {}
    for name, waveform in clips.items():
        inputs = processor(audio=waveform, sampling_rate=SAMPLE_RATE)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, return_dict_in_generate=True)
        text, offsets = processor.decode(output.sequences, durations=output.durations, skip_special_tokens=True)
        feature_count = int(inputs["attention_mask"][0].sum())
        if name == "speech":
            np.save(feature_path, inputs["input_features"][0, :feature_count].numpy())
        results[name] = {
            "text": text[0],
            "input_shape": list(inputs["input_features"].shape),
            "feature_frames": feature_count,
            "token_times": offsets[0],
            "token_ids": output.sequences[0].tolist(),
            "step_advances": output.durations[0].tolist(),
        }
    return {
        "window_samples": extractor.win_length,
        "hop_samples": extractor.hop_length,
        "fft_size": extractor.n_fft,
        "mel_bins": extractor.feature_size,
        "encoder_subsampling": encoder_config.subsampling_factor,
        "encoder_hidden_size": encoder_config.hidden_size,
        "encoder_step_seconds": extractor.hop_length * encoder_config.subsampling_factor / SAMPLE_RATE,
        "duration_choices": model.config.durations,
        "blank_id": model.config.blank_token_id,
        "clips": results,
    }


def transcribe_qwen(path, clips):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        path, dtype=torch.float32, use_safetensors=True, local_files_only=True,
    ).eval()
    results = {}
    for name, waveform in clips.items():
        inputs = processor.apply_transcription_request(audio=waveform, return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=128)
        generated = output[:, inputs["input_ids"].shape[1]:]
        parsed = processor.decode(generated, return_format="parsed")[0]
        results[name] = {
            "text": parsed["transcription"],
            "language": parsed["language"],
            "raw_text": processor.decode(generated)[0],
            "feature_shape": list(inputs["input_features"].shape),
            "generated_ids": generated[0].tolist(),
        }
    return {"clips": results}


def tag_melati(path, clips, feature_path):
    import torch
    from safetensors.torch import load_file
    from torchaudio.compliance.kaldi import fbank
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # Use the verified architecture and require a strict match to the saved tensors.
    model_type = get_class_from_dynamic_module("modeling_eat.EATModel", str(path), local_files_only=True)
    config = model_type.config_class.from_pretrained(path, local_files_only=True)
    model = model_type(config).eval()
    model.load_state_dict(load_file(path / "model.safetensors"), strict=True)
    label_ids = {label: i for i, label in model.config.id2label.items()}
    tracked = ("Speech", "Dog", "Bark", "Rain", "Silence")
    results = {}
    for name, waveform in clips.items():
        wave = torch.from_numpy(waveform).unsqueeze(0)
        raw = fbank(
            wave - wave.mean(), htk_compat=True, sample_frequency=SAMPLE_RATE,
            use_energy=False, window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10,
        )
        fixed = torch.nn.functional.pad(raw, (0, 0, 0, max(0, 1024 - len(raw))))[:1024]
        inputs = ((fixed + 4.268) / (4.569 * 2))[None, None]
        with torch.inference_mode():
            logits = model(inputs)[0]
        scores = logits.sigmoid()
        top_indices = scores.topk(5).indices.tolist()
        if name == "speech-dog":
            np.save(feature_path, inputs[0, 0].numpy())

        def score(i):
            return {"id": i, "label": model.config.id2label[i],
                    "logit": logits[i].item(), "sigmoid": scores[i].item()}

        results[name] = {
            "frames_before_padding": len(raw),
            "input_shape": list(inputs.shape),
            "top_five": [score(i) for i in top_indices],
            "selected": {label: score(label_ids[label]) for label in tracked},
        }
    return {"num_labels": model.config.num_classes, "patch_size": model.config.patch_size,
            "patch_stride": model.config.stride, "clips": results}


def run(output_dir, audio_dir, model_names, offline, threads):
    import torch

    torch.set_num_threads(threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((audio_dir / "manifest.json").read_text())
    clips = {}
    for name in CLIPS:
        path = audio_dir / f"{name}.wav"
        if digest(path) != manifest["clips"][name]["sha256"]:
            raise ValueError(f"Unexpected PCM fixture: {path}")
        clips[name] = load_wave(path).astype(np.float32)
    clips["silence"] = np.zeros_like(clips["rain"])
    report = {
        "audio": manifest,
        "runtime": {name: version(name) for name in ("torch", "torchaudio", "transformers", "timm", "numpy")},
        "device": "cpu",
        "inference_dtype": "float32",
        "threads": threads,
        "models": {},
    }
    for name in model_names:
        print(f"Running {name} on {len(clips)} clips", flush=True)
        path = load_checkpoint(name, offline)
        if name == "parakeet":
            result = transcribe_parakeet(path, clips, output_dir / "parakeet_input.npy")
        elif name == "qwen":
            result = transcribe_qwen(path, clips)
        else:
            result = tag_melati(path, clips, output_dir / "melati_input.npy")
        if name in ("parakeet", "qwen"):
            for clip, record in result["clips"].items():
                reference = manifest["reference"] if clip.startswith("speech") else ""
                record["reference_words"] = words(reference)
                record["hypothesis_words"] = words(record["text"])
                record["errors"] = word_error_rate(record["reference_words"], record["hypothesis_words"])
        repository, revision, weights = CHECKPOINTS[name]
        report["models"][name] = {"repository": repository, "revision": revision,
                                  "weights_sha256": weights, **result}
        (output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Use only the cached pinned snapshots")
    parser.add_argument("--model", choices=CHECKPOINTS, action="append")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.output_dir, args.audio_dir, args.model or CHECKPOINTS, args.offline, args.threads)
