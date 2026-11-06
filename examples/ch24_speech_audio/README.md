# Chapter 24. Speech and Audio

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch24_speech_audio.run_examples --output output/ch24.json
python -m unittest discover -s tests -p test_speech_audio.py -v
```

## Inspect short frames of a waveform

`frame_power_spectrogram` takes one real sample vector, a sample rate,
a frame length, and a hop length. It reads only full frames. For each
frame it multiplies samples by a symmetric Hann window and squares
the magnitude of the real FFT. The result stores frame rows and
nonnegative-frequency columns. It also returns the start time of each
frame and the frequency of each bin. These are raw squared magnitudes.
For an input shorter than one frame, the power array has zero rows.

## Label generated tones

The example generates one clean and one noisy clip at each of 500 Hz
and 1000 Hz, plus one silent clip. All five clips contain 1024 samples
at 8000 Hz. The fixed frame and hop lengths yield seven complete frames.
`tone_label` uses the waveform-wide RMS threshold of 0.001 for silence.
For other clips it averages power over frames, finds the peak frequency
outside DC, and returns the nearer of the two known tone labels. This
experiment classifies generated signals. The word-error example below
uses supplied word sequences.

## Count word edits

`word_error_rate` accepts pretokenized reference and hypothesis
words. Set case, punctuation, and whitespace rules before tokenizing
both sequences. The function uses exact word equality and selects a
minimum-edit path. For ties it prefers the diagonal step, then a
deletion, then an insertion. Different minimum paths can have the same
total cost.

The report includes substitutions, deletions, insertions, reference
length, and their normalized sum. WER can exceed one. When the
reference has no words, the function still reports the insertions and
sets the rate to `None`.
