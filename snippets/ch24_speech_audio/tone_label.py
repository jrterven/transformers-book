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
