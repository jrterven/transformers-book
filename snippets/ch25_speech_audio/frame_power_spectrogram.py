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
