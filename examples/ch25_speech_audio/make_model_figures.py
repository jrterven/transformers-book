"""Render the audio figures from a saved model run and its PCM fixtures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Rectangle
import numpy as np

from .model import frame_power_spectrogram
from .prepare_audio import AUDIO_DIR, SAMPLE_RATE, load_wave


INK = "#17334b"
ACCENT = "#14796f"
FONT = "DejaVu Sans"
SPEECH_ORDER = ("speech", "speech-rain-0db", "speech-rain-minus10db")
TAG_ORDER = ("speech", "dog", "speech-dog", "rain", "speech-rain-0db", "speech-rain-minus10db", "silence")
CLIP_NAMES = {
    "speech": "Clean voice",
    "dog": "Dog",
    "speech-dog": "Voice + dog",
    "rain": "Rain",
    "speech-rain-0db": "Voice + rain   |   0 dB",
    "speech-rain-minus10db": "Voice + rain   |   −10 dB",
    "silence": "Digital silence",
}


def save(figure, output_dir, name):
    raster = name in {"parakeet_frontend", "rain_spectrograms", "melati_patches"}
    suffix = "jpg" if raster else "png"
    options = {"pil_kwargs": {"quality": 91, "subsampling": 1, "optimize": True}} if raster else {}
    figure.savefig(output_dir / f"ch25_{name}.{suffix}", dpi=150 if raster else 180, facecolor="white",
                   bbox_inches="tight", pad_inches=0.09, **options)
    plt.close(figure)


def clean_frontend(speech, features, hop, output):
    fig, (wave_ax, spec_ax) = plt.subplots(
        2, 1, figsize=(9, 4.25), sharex=True, height_ratios=(1, 1.72), layout="constrained",
    )
    time = np.arange(len(speech)) / SAMPLE_RATE
    wave_ax.plot(time, speech, color=ACCENT, lw=0.55, rasterized=True)
    wave_ax.axhline(0, color=INK, lw=0.45, alpha=0.5)
    wave_ax.set_title("Recorded waveform", loc="left")
    wave_ax.set_ylabel("Amplitude")
    vmin, vmax = np.percentile(features, [1, 99])
    image = spec_ax.imshow(features.T, extent=(-hop / 2, (len(features) - 0.5) * hop, 0, features.shape[1]),
                           aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    spec_ax.set_title("Input to the Parakeet encoder", loc="left")
    spec_ax.set(xlabel="Time (s)", ylabel="Mel band", xlim=(0, len(speech) / SAMPLE_RATE),
                yticks=[0, 32, 64, 96, 128])
    fig.colorbar(image, ax=(wave_ax, spec_ax), shrink=0.73, label="Normalized feature")
    save(fig, output, "parakeet_frontend")


def rain_spectrograms(clips, output):
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.55), sharex=True, sharey=True, layout="constrained")
    spectra = [frame_power_spectrogram(clips[name], SAMPLE_RATE, 400, 160) for name in SPEECH_ORDER]
    reference = max(power.max() for _, _, power in spectra)
    for axis, name, (starts, frequencies, power) in zip(axes, SPEECH_ORDER, spectra):
        db = 10 * np.log10(np.maximum(power / reference, 1e-7))
        image = axis.pcolormesh(starts + 200 / SAMPLE_RATE, frequencies / 1000, db.T,
                                shading="nearest", vmin=-70, vmax=0, cmap="magma", rasterized=True)
        axis.set_title(CLIP_NAMES[name], loc="left")
        axis.set(ylabel="kHz", yticks=(0, 2, 4, 6, 8), ylim=(0, 8),
                 xlim=(0, len(clips[name]) / SAMPLE_RATE))
    axes[-1].set_xlabel("Time (s)")
    fig.colorbar(image, ax=axes, shrink=0.95, label="Power relative to shared peak (dB)")
    save(fig, output, "rain_spectrograms")


def tdt_trace(speech, parakeet, output):
    clip = parakeet["clips"]["speech"]
    step_seconds = parakeet["encoder_step_seconds"]
    advances = np.array(clip["step_advances"][1:9])
    ids = clip["token_ids"][1:9]
    starts = np.r_[0, np.cumsum(advances[:-1])]
    end = float(advances.sum() * step_seconds)
    emitted = iter(clip["token_times"])
    fig, (wave, path) = plt.subplots(2, 1, figsize=(9, 3.15), sharex=True,
                                     height_ratios=(1.6, 1), layout="constrained")
    samples = round(end * SAMPLE_RATE)
    wave.plot(np.arange(samples) / SAMPLE_RATE, speech[:samples], color=ACCENT, lw=0.65)
    wave.set_title("Start of the clean recording", loc="left")
    wave.set_ylabel("Amplitude")
    wave.set_xlim(0, end)
    for start, width, token in zip(starts, advances, ids):
        blank = token == parakeet["blank_id"]
        left = start * step_seconds
        span = width * step_seconds
        label = "Blank" if blank else next(emitted)["token"].strip()
        wave.axvline(left, color=INK, lw=0.75, alpha=0.22)
        path.add_patch(Rectangle((left, 0.22), span, 0.67, facecolor="#e2e9ef" if blank else "#c3e9df",
                                 edgecolor="white", linewidth=1.5))
        path.text(left + span / 2, 0.55, label, ha="center", va="center")
    path.set_title("Decoded advances on the encoder grid", loc="left")
    path.set(xlabel="Time (s)", yticks=[], ylim=(0, 1))
    ticks = np.r_[starts, advances.sum()]
    path.set_xticks(ticks * step_seconds, [f"{i * step_seconds:.2f}" for i in ticks])
    path.tick_params(axis="x", length=3)
    path.spines["left"].set_visible(False)
    save(fig, output, "tdt_trace")


def melati_patches(features, padded_after, stride, output):
    fig, (full, zoom) = plt.subplots(2, 1, figsize=(9, 4.35), layout="constrained")
    band_count = features.shape[1]
    hop = 0.010
    first = 6 * stride
    stop = 12 * stride
    vmin, vmax = np.percentile(features[:padded_after], [1, 99])
    for axis, start, end in ((full, 0, len(features)), (zoom, first, stop)):
        image = axis.imshow(features[start:end].T, extent=(start * hop, end * hop, 0, band_count),
                            origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        axis.set(ylabel="Mel band", yticks=(0, 32, 64, 96, 128))
    full.set_title("Feature array supplied to the classifier", loc="left")
    full.axvline(padded_after * hop, color="white", ls="--", lw=1.25)
    full.text(padded_after * hop + 0.14, 108, "Padding", color=INK,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    zoom.set_title("A closer look at six time patches", loc="left")
    for boundary in range(first + stride, stop, stride):
        zoom.axvline(boundary * hop, color="white", alpha=0.70, lw=0.8)
    for boundary in range(stride, band_count, stride):
        zoom.axhline(boundary, color="white", alpha=0.70, lw=0.8)
    zoom.set_xticks(np.arange(first, stop + 1, stride) * hop)
    zoom.set_xlabel("Time (s) from frame starts")
    fig.colorbar(image, ax=(full, zoom), shrink=0.87, label="Normalized feature")
    save(fig, output, "melati_patches")


def melati_scores(results, output):
    labels = ["Speech", "Dog", "Bark", "Rain"]
    scores = np.array([[results[name]["selected"][label]["sigmoid"] for label in labels]
                       for name in TAG_ORDER])
    fig, axis = plt.subplots(figsize=(7.6, 4.25), layout="constrained")
    color = colormaps["Blues"]
    image = axis.imshow(scores, cmap=color, vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(labels)), labels)
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(TAG_ORDER)), [CLIP_NAMES[key] for key in TAG_ORDER])
    for row, values in enumerate(scores):
        for column, value in enumerate(values):
            label = f"{value:.3f}" if value >= 0.001 else "<0.001"
            axis.text(column, row, label, ha="center", va="center",
                      color="white" if value >= 0.64 else INK,
                      fontweight="bold" if value >= 0.5 else "normal")
    axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(TAG_ORDER), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.4)
    axis.tick_params(which="both", length=0)
    fig.colorbar(image, ax=axis, shrink=0.94, label="Sigmoid score")
    save(fig, output, "melati_scores")


def main(audio_dir, run_dir, output_dir):
    plt.rcParams.update({
        "font.family": FONT, "font.size": 11.5, "axes.labelcolor": INK, "text.color": INK,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": INK, "ytick.color": INK,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads((run_dir / "results.json").read_text())["models"]
    clips = {name: load_wave(audio_dir / f"{name}.wav") for name in SPEECH_ORDER}
    parakeet = results["parakeet"]
    clean_frontend(clips["speech"], np.load(run_dir / "parakeet_input.npy"),
                   parakeet["hop_samples"] / SAMPLE_RATE, output_dir)
    rain_spectrograms(clips, output_dir)
    tdt_trace(clips["speech"], parakeet, output_dir)
    melati = results["melati"]
    melati_patches(np.load(run_dir / "melati_input.npy"),
                   melati["clips"]["speech-dog"]["frames_before_padding"],
                   melati["patch_stride"], output_dir)
    melati_scores(melati["clips"], output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.audio_dir, args.run_dir, args.output_dir)
