"""Build the small PCM fixtures from the three credited source WAV files."""

import argparse
from hashlib import file_digest
import json
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


SAMPLE_RATE = 16_000
AUDIO_DIR = Path(__file__).with_name("audio")
REFERENCE = "Por ello, cuando se inventó, el lápiz ganó muchos amigos."
SOURCE_FILES = {
    "speech": "15635660839838523367.wav",
    "rain": "1-21189-A-10.wav",
    "dog": "1-100032-A-0.wav",
}
CLIPS = ("speech", "speech-rain-0db", "speech-rain-minus10db", "rain", "dog", "speech-dog")


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def load_wave(path):
    rate, pcm = wavfile.read(path)
    samples = pcm.astype(np.float64)
    if np.issubdtype(pcm.dtype, np.signedinteger):
        samples /= -np.iinfo(pcm.dtype).min
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    divisor = gcd(rate, SAMPLE_RATE)
    if rate != SAMPLE_RATE:
        samples = resample_poly(samples, SAMPLE_RATE // divisor, rate // divisor)
    return samples


def rms(samples):
    return float(np.sqrt(np.mean(samples**2)))


def write_pcm(path, samples):
    pcm = np.rint(samples * 32768).astype("<i2")
    wavfile.write(path, SAMPLE_RATE, pcm)
    return {
        "samples": len(pcm),
        "seconds": len(pcm) / SAMPLE_RATE,
        "sha256": digest(path),
        "rms": rms(pcm.astype(np.float64) / 32768),
    }


def prepare(source_dir, output_dir):
    source_paths = {name: source_dir / path for name, path in SOURCE_FILES.items()}
    speech, rain, dog = (load_wave(source_paths[name]) for name in ("speech", "rain", "dog"))
    # Extend the rain smoothly to the utterance length. Place the dog clip
    # 400 ms into the utterance, with zeros outside the clip.
    rain_track = np.pad(rain, (0, len(speech) - len(rain)), mode="reflect")
    dog_start = round(0.4 * SAMPLE_RATE)
    dog_track = np.pad(dog, (dog_start, len(speech) - len(dog) - dog_start))
    mixes = {
        "speech-rain-0db": (rain_track, 0),
        "speech-rain-minus10db": (rain_track, -10),
        "speech-dog": (dog_track, 3),
    }
    added = {}
    for name, (noise, snr_db) in mixes.items():
        gain = rms(speech) / (rms(noise) * 10 ** (snr_db / 20))
        added[name] = noise * gain
    # Use one gain on the four speech clips so their amplitudes are comparable.
    speech_clips = {"speech": speech, **{name: speech + noise for name, noise in added.items()}}
    shared_gain = min(1.0, 0.90 / max(np.max(np.abs(x)) for x in speech_clips.values()))
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = {
        name: write_pcm(output_dir / f"{name}.wav", shared_gain * samples)
        for name, samples in speech_clips.items()
    }
    for name, noise in added.items():
        clips[name]["component_snr_db"] = 20 * np.log10(rms(speech) / rms(noise))
    for name, samples in {"rain": rain, "dog": dog}.items():
        clips[name] = write_pcm(output_dir / f"{name}.wav", samples * (0.90 / np.max(np.abs(samples))))
    manifest = {
        "sample_rate": SAMPLE_RATE,
        "reference": REFERENCE,
        "sources": {name: {"file": path.name, "sha256": digest(path)} for name, path in source_paths.items()},
        "speech_gain": float(shared_gain),
        "clips": clips,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output", type=Path, default=AUDIO_DIR)
    args = parser.parse_args()
    prepare(args.source_dir, args.output)
