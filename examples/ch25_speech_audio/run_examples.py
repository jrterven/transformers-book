"""Run offline synthetic audio and word-error examples."""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import frame_power_spectrogram, tone_label, word_error_rate


def short_frame_example():
    times, frequencies, power = frame_power_spectrogram([0, 1, 1, 0], 8, 4, 2)
    return {
        "frame_start_seconds": times.tolist(),
        "frequencies_hz": frequencies.tolist(),
        "raw_power": np.round(power, 12).tolist(),
    }


def tone_example():
    sample_rate, frame_length, hop_length = 8000, 256, 128
    samples = 1024
    t = np.arange(samples) / sample_rate
    rng = np.random.default_rng(24)
    clips = []
    for frequency in (500, 1000):
        clean = 0.6 * np.sin(2 * np.pi * frequency * t)
        noisy = clean + rng.normal(0, 0.15, size=samples)
        clips.extend(((f"clean_{frequency}", clean, frequency),
                      (f"noisy_{frequency}", noisy, frequency)))
    clips.append(("silence", np.zeros(samples), "silence"))
    times, frequencies, power = frame_power_spectrogram(
        clips[0][1], sample_rate, frame_length, hop_length
    )
    results = {}
    for name, waveform, target in clips:
        label, peak_hz, rms = tone_label(
            waveform, sample_rate, frame_length, hop_length
        )
        results[name] = {
            "target": target,
            "label": label,
            "peak_hz": peak_hz,
            "rms": round(rms, 6),
        }
    return {
        "sample_rate": sample_rate,
        "frame_length": frame_length,
        "hop_length": hop_length,
        "samples_per_clip": samples,
        "power_shape": list(power.shape),
        "frame_start_seconds": times.tolist(),
        "frequency_step_hz": float(frequencies[1] - frequencies[0]),
        "clips": results,
    }


def wer_example():
    reference = "we heard blue birds sing".split()
    hypothesis = "we saw birds sing today".split()
    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "scores": word_error_rate(reference, hypothesis),
        "empty_reference": word_error_rate([], ["hello"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {
        "short_frame": short_frame_example(),
        "generated_tones": tone_example(),
        "word_error_rate": wer_example(),
    }
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
