"""Check Chapter 24 framing, generated tones, and word-error alignment."""

import math
import unittest

import numpy as np

from examples.ch24_speech_audio.model import (
    frame_power_spectrogram,
    tone_label,
    word_error_rate,
)
from examples.ch24_speech_audio.run_examples import short_frame_example, tone_example, wer_example


class SpectrogramTests(unittest.TestCase):
    def test_worked_four_samples_and_raw_power(self):
        self.assertEqual(short_frame_example(), {
            "frame_start_seconds": [0],
            "frequencies_hz": [0, 2, 4],
            "raw_power": [[2.25, 1.125, 0]],
        })

    def test_complete_odd_frames_match_explicit_dft(self):
        waveform = np.arange(11, dtype=float)
        sample_rate, frame_length, hop = 10, 5, 4
        times, frequencies, power = frame_power_spectrogram(
            waveform, sample_rate, frame_length, hop
        )
        np.testing.assert_array_equal(times, [0, 0.4])
        np.testing.assert_array_equal(frequencies, [0, 2, 4])
        expected = []
        for start in (0, 4):
            row = []
            for k in range(3):
                total = sum(
                    waveform[start + j] * (0.5 - 0.5 * math.cos(2 * math.pi * j / 4))
                    * np.exp(-2j * np.pi * j * k / 5)
                    for j in range(frame_length)
                )
                row.append(abs(total) ** 2)
            expected.append(row)
        np.testing.assert_allclose(power, expected, atol=1e-12, rtol=0)
        waveform[9:] += 100
        _, _, changed = frame_power_spectrogram(waveform, sample_rate, frame_length, hop)
        np.testing.assert_array_equal(changed, power)

    def test_empty_frame_axis_and_zero_power(self):
        times, frequencies, power = frame_power_spectrogram([1, 2], 8, 4, 2)
        self.assertEqual(times.shape, (0,))
        np.testing.assert_array_equal(frequencies, [0, 2, 4])
        self.assertEqual(power.shape, (0, 3))
        _, _, silence = frame_power_spectrogram(np.zeros(16), 8000, 16, 8)
        np.testing.assert_array_equal(silence, np.zeros((1, 9)))

    def test_invalid_waveform_and_frame_arguments(self):
        with self.assertRaisesRegex(ValueError, "one vector"):
            frame_power_spectrogram([[1, 2]], 8, 4, 2)
        with self.assertRaisesRegex(ValueError, "finite"):
            frame_power_spectrogram([1, np.inf], 8, 4, 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            frame_power_spectrogram([1, 2], 8, 4, 0)

    def test_generated_clean_noisy_and_silent_clips(self):
        result = tone_example()
        self.assertEqual(result["power_shape"], [7, 129])
        self.assertEqual(result["frequency_step_hz"], 31.25)
        for clip in result["clips"].values():
            self.assertEqual(clip["label"], clip["target"])
        self.assertEqual(result["clips"]["noisy_500"]["peak_hz"], 500)
        self.assertEqual(result["clips"]["noisy_1000"]["peak_hz"], 1000)
        self.assertIsNone(result["clips"]["silence"]["peak_hz"])

    def test_silence_threshold_applies_to_the_whole_clip(self):
        waveform = np.zeros(1024)
        waveform[128] = 0.02
        label, frequency, rms = tone_label(waveform, 8000, 256, 128)
        self.assertEqual((label, frequency), ("silence", None))
        self.assertAlmostEqual(rms, 0.02 / math.sqrt(len(waveform)))
        with self.assertRaisesRegex(ValueError, "complete frame"):
            tone_label([0], 8000, 256, 128)


class WordErrorTests(unittest.TestCase):
    def test_worked_three_kinds_of_edits(self):
        self.assertEqual(wer_example()["scores"], {
            "substitutions": 1, "deletions": 1, "insertions": 1,
            "edits": 3, "reference_words": 5, "wer": 0.6,
        })

    def test_ties_prefer_diagonal_edits(self):
        result = word_error_rate(["a", "b"], ["b", "a"])
        self.assertEqual(result["substitutions"], 2)
        self.assertEqual(result["deletions"], 0)
        self.assertEqual(result["insertions"], 0)

    def test_empty_reference_and_rates_above_one(self):
        self.assertIsNone(word_error_rate([], [])["wer"])
        empty_reference = word_error_rate([], ["unexpected", "words"])
        self.assertIsNone(empty_reference["wer"])
        self.assertEqual(empty_reference["insertions"], 2)
        self.assertEqual(word_error_rate(["a", "b"], [])["deletions"], 2)
        self.assertEqual(word_error_rate(["a"], ["a", "b", "c"])["wer"], 2)

    def test_word_equality_uses_supplied_tokens(self):
        self.assertEqual(word_error_rate(["Blue"], ["blue"])["substitutions"], 1)
        self.assertEqual(word_error_rate(["blue"], ["blue"])["wer"], 0)
        with self.assertRaisesRegex(TypeError, "pretokenized"):
            word_error_rate("raw words", ["raw", "words"])


if __name__ == "__main__":
    unittest.main()
