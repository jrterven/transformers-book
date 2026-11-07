# Chapter 24 exercises

Speech and Audio

Exercises 1-4 work through samples, frames, tasks and edits by hand.
Exercises 5-8 use `examples.ch24_speech_audio.model`.
Exercises 9-11 design evaluations for new speakers, no-speech clips and
live captions. All eleven have [complete solutions](../../solutions/ch24_speech_audio/README.md).

1. **Count complete frames.** A recording has 24 samples. Use a frame length of eight and a hop of four. List the frame start indices and state how many samples consecutive frames share. Repeat with hop eight.

2. **Read the axes.** For $f_s=8{,}000$ Hz, $N=16$, and $H=8$, write the frequencies of bins 0, 1, and 8 and the spacing between frame starts. Explain which of $N$ and $H$ sets each spacing. Give the window duration. Change only $N$ to 32, then only $H$ to four, and state what each change affects.

3. **Choose a decoding task.** A clip contains the words *buenos días*. Specify the language and task choices for a Spanish transcript and for an English translation. Explain what stays the same in the encoder input.

4. **Align the words.** Use reference *turn at the blue door* and hypothesis *turn at the red door now*. Mark a minimum-edit alignment and compute its WER. Explain how the denominator changes if you reverse the two sentences.

5. **Check the spectrogram shape.** Call `frame_power_spectrogram` with 16 zero samples, sample rate 8{,}000, frame length eight, and hop four. Unpack the frame start times, bin frequencies, and power. Verify the axes and the shape from Section 24.2. Check every power value.

6. **Find a tone.** Generate 32 samples of $\sin(2\pi\cdot 1{,}000 n/8{,}000)$ for $n=0,\ldots,31$. Use sample rate 8{,}000, frame length 16, and hop eight. Locate the maximum-power bin in each frame. Repeat with a 2{,}000 Hz tone and relate the bin numbers to the frequencies.

7. **Compare clip and corpus errors.** Pass word lists to `word_error_rate`. Use reference *go home* with hypothesis *go*, then use *please turn left now* on both sides. Compute pooled WER from `edits` and `reference_words`. Compare it with the mean of the two returned `wer` values. Add an empty reference with hypothesis *thank you*. Recompute pooled WER and explain why the new clip has no individual WER.

8. **Audit a tie.** Use the same function with reference *a b* and hypothesis *b a*. Write down two minimum-edit alignments with different substitution, deletion, or insertion counts. Identify which counts the implementation returns and how its tie rule explains them.

9. **Check noise without leakage.** You have recordings from 30 speakers, each in clean and noisy versions. Design training, validation, and test splits for new speakers. Explain where you put both versions of each recording and which subset scores you report.

10. **Score false words.** Design a no-speech evaluation with silent clips and clips containing non-speech sound. Choose a clip-level measure of false words, then state how insertions from these clips can enter a pooled WER with spoken clips.

11. **Measure live captions.** Two systems have the same offline WER. One updates words sooner and frequently revises them. Design a timing protocol that measures both first appearance and stabilization. Describe the timestamps you need and the hardware and buffering settings you would record. For a word ending at 1.20 s and displayed as *turn*, *term*, and *turn* at 1.35, 1.48, and 1.62 s, compute both latencies. Explain how to report a word that never becomes correct.
