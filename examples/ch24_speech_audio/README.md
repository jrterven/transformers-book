# Chapter 24. Speech and Audio

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch24_speech_audio.run_examples --output output/ch24.json
python -m unittest discover -s tests -p test_speech_audio.py -v
```

## Listen to the model inputs

Every file is mono, 16 kHz, 16-bit PCM. The three speech conditions have
the same voice track and the same gain on it. The audio and credits are
published together. The licenses below apply to the recordings and their
derivatives.

| Recording | Length | Audio |
| --- | ---: | --- |
| Clean Latin American Spanish | 5.64 s | [Listen](audio/speech.wav) |
| Same voice with rain at 0 dB SNR | 5.64 s | [Listen](audio/speech-rain-0db.wav) |
| Same voice with rain at −10 dB SNR | 5.64 s | [Listen](audio/speech-rain-minus10db.wav) |
| Rain alone | 5.00 s | [Listen](audio/rain.wav) |
| Dog alone | 5.00 s | [Listen](audio/dog.wav) |
| Same voice with a dog at +3 dB SNR | 5.64 s | [Listen](audio/speech-dog.wav) |

- The voice comes from [Google FLEURS](https://huggingface.co/datasets/google/fleurs),
  revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, Latin American Spanish
  (`es_419`), test example 1880, WAV `15635660839838523367.wav`.
  The published reference is “Por ello, cuando se inventó, el lápiz ganó muchos amigos.”
  We applied gain and converted to 16-bit PCM. The variants add the indicated
  background. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- The rain is `1-21189-A-10.wav` from Karol J. Piczak's ESC-10 subset of
  [ESC-50](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6),
  derived from [Louisiana Rain 1.wav by uberhuberman](https://freesound.org/people/uberhuberman/sounds/21189/)
  (CC0). We resampled and adjusted its gain. In the mixtures we extended the
  5 s rain track by reflection for 0.64 s. [ESC-10 license, CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)
- The dog is `1-100032-A-0.wav` from the same ESC-10 release, derived from
  [rose_bark.wav by nfrae](https://freesound.org/people/nfrae/sounds/100032/)
  (CC0 per the [dataset credits](https://github.com/karolpiczak/ESC-50/blob/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/LICENSE)).
  We resampled and adjusted its gain. In the mixture we start the dog track
  0.40 s into the voice clip. [ESC-10 license, CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)

The SNRs refer to the voice and the added track over the entire clip before
rounding to PCM. The run also creates a 5 s all-zero array in memory.
The source names, audio hashes, and measured gains are in
[the audio manifest](audio/manifest.json).

## Run the pretrained demonstrations

Use Python 3.12. The three pinned checkpoint files total about 5.3 GB.
The commands run on CPU and load one model at a time.

```sh
python -m pip install -r requirements-ch24-figures.txt
python -m examples.ch24_speech_audio.run_model_demos --output-dir output/ch24-models
python -m examples.ch24_speech_audio.make_model_figures \
  --run-dir output/ch24-models --output-dir output/ch24-figures
```

The model runner checks the checkpoint SHA-256 hashes. It loads the four
architecture files that define Melati from the pinned local snapshot and
verifies their Git blob IDs before importing them. Use `--offline` after
the snapshots are cached. Use `--model parakeet`, `--model qwen`, or
`--model melati` to run a single model.

[The recorded results](model_results.json) contain the checkpoint revisions,
raw transcripts, normalized word edits, TDT advances, and selected logits
and sigmoid scores. The spectrogram script reads the same WAV files and the
model input arrays saved by the runner. For the unpadded STFT comparison it
uses one power scale across all three speech conditions.

To rebuild the WAV fixtures, place the three source files named in the
manifest in one directory. The Spanish WAV is in the `es_419` test archive
of FLEURS, and the other two are in the ESC-50 `audio` directory. Run
`python -m examples.ch24_speech_audio.prepare_audio SOURCE_DIR --output output/ch24-audio`.

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

## Follow the units and denominators

A sample is one amplitude measurement; a frame contains several samples;
a text token is a decoder output. Window duration is `N / sample_rate`,
frame-start spacing is `H / sample_rate`, and Fourier-bin spacing is
`sample_rate / N`. More overlapping frames do not shorten the window.
Changing a sample-rate label is not resampling the signal.

The helper uses complete unpadded frames and a symmetric Hann window.
It returns raw one-sided squared FFT magnitudes without power-density
normalization. A log-Mel frontend also combines frequency bins with a
filter bank. The toy RMS threshold handles near-zero amplitudes, not
general voice detection, and any non-silent input is assigned to one of
two allowed tone labels.

Pooled WER adds edits and reference words before division. A no-speech
clip can add insertions without adding denominator words. Report its
false-word behavior separately. For live captions, measure first correct
appearance and correct stabilization, and report how many reference words
have each latency; missing words cannot be hidden by a latency average.

See the [11 exercises](EXERCISES.md) and [complete solutions](../../solutions/ch24_speech_audio/README.md).
