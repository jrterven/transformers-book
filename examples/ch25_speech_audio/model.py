"""NumPy teaching calculations for speech and audio features."""

from operator import index

import numpy as np


def _waveform(samples):
    samples = np.asarray(samples)
    if samples.ndim != 1 or samples.dtype.kind not in "iuf":
        raise ValueError("waveform needs one vector of real samples")
    samples = samples.astype(np.float64, copy=False)
    if not np.isfinite(samples).all():
        raise ValueError("waveform samples must be finite")
    return samples


# snippet:frame_power_spectrogram
def frame_power_spectrogram(waveform, sample_rate, frame_length, hop_length):
    """Return frame start times, frequency bins, and raw one-sided power."""
    waveform = _waveform(waveform)
    sample_rate, frame_length, hop_length = map(
        index, (sample_rate, frame_length, hop_length)
    )
    if min(sample_rate, frame_length, hop_length) < 1:
        raise ValueError("sample rate, frame length, and hop length must be positive")

    starts = np.arange(0, len(waveform) - frame_length + 1, hop_length)
    times = starts / sample_rate
    frequencies = np.fft.rfftfreq(frame_length, d=1 / sample_rate)
    if len(starts) == 0:
        return times, frequencies, np.empty((0, len(frequencies)))

    sample_indices = starts[:, None] + np.arange(frame_length)
    frames = waveform[sample_indices]
    windowed = frames * np.hanning(frame_length)
    with np.errstate(over="raise", invalid="raise"):
        spectrum = np.fft.rfft(windowed, axis=-1)
        power = np.abs(spectrum) ** 2
    return times, frequencies, power
# end-snippet:frame_power_spectrogram


# snippet:tone_label
def tone_label(waveform, sample_rate, frame_length, hop_length):
    """Return the label, non-DC peak frequency, and RMS for a toy clip."""
    waveform = _waveform(waveform)
    _, frequencies, power = frame_power_spectrogram(
        waveform, sample_rate, frame_length, hop_length
    )
    if power.shape[0] == 0 or power.shape[1] < 2:
        raise ValueError("tone labels need a complete frame and a non-DC bin")
    peak_amplitude = np.max(np.abs(waveform))
    if peak_amplitude == 0:
        return "silence", None, 0.0
    scaled = waveform / peak_amplitude
    rms = float(peak_amplitude * np.sqrt(np.mean(scaled**2)))
    if rms < 0.001:
        return "silence", None, rms
    peak = int(power.mean(axis=0)[1:].argmax()) + 1
    peak_hz = float(frequencies[peak])
    label = min((500, 1000), key=lambda candidate: abs(candidate - peak_hz))
    return label, peak_hz, rms
# end-snippet:tone_label


def _words(tokens):
    if isinstance(tokens, (str, bytes)):
        raise TypeError("pass pretokenized word sequences")
    tokens = tuple(tokens)
    if any(not isinstance(word, str) or not word or word.isspace() for word in tokens):
        raise ValueError("word tokens must be nonempty strings")
    return tokens


# snippet:word_error_rate
def word_error_rate(reference, hypothesis):
    """Find minimum edit counts and divide by the reference word count."""
    reference, hypothesis = map(_words, (reference, hypothesis))
    # Store (substitutions, deletions, insertions) for each prefix pair.
    previous = [(0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, actual in enumerate(reference, start=1):
        current = [(0, i, 0)]
        for j, guess in enumerate(hypothesis, start=1):
            s, d, ins = previous[j - 1]
            diagonal = (s + int(actual != guess), d, ins)
            s, d, ins = previous[j]
            deletion = (s, d + 1, ins)
            s, d, ins = current[-1]
            insertion = (s, d, ins + 1)
            # Ties prefer the diagonal, then deletion, then insertion.
            current.append(min((diagonal, deletion, insertion), key=sum))
        previous = current

    substitutions, deletions, insertions = previous[-1]
    edits = substitutions + deletions + insertions
    return {
        "substitutions": substitutions, "deletions": deletions,
        "insertions": insertions, "edits": edits,
        "reference_words": len(reference),
        "wer": edits / len(reference) if reference else None,
    }
# end-snippet:word_error_rate
