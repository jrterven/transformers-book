# Chapter 24: complete worked solutions

All [11 exercises](../../examples/ch24_speech_audio/EXERCISES.md) are solved
step by step in each edition:

- [Written solutions](written-solutions.pdf): explanations, calculations, tables and three plots.
- [Executed notebook](python-solutions.ipynb): every explanation with 16 executed cells.
- [Notebook PDF](python-solutions.pdf): all prose, code, figures and saved results.

The plots show overlapping frame intervals, two tones' spectra and a
caption's first-correct and stabilization times. The synthetic fixtures
also audit speaker grouping and false words on no-speech clips. They are
teaching calculations, not accuracy measurements of a trained recognizer.

## Run and reproduce

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch24_speech_audio/python-solutions.ipynb`, restart the kernel
and run all cells in order. The saved run used Python 3.12.14, NumPy 2.5.4
and Matplotlib 3.11.2 on CPU. No model or dataset download is needed.

To rebuild both PDFs from the same executed notebook, install Tectonic
or XeLaTeX and run:

```sh
python -m pip install mistune
python solutions/ch24_speech_audio/build_pdfs.py
```

An explicit engine path can be supplied with `--engine`. The exporter
checks all eleven exercise headings, sequential execution, recorded results
for every exercise and no execution errors.

## Canonical example checks

```sh
python -m unittest discover -s tests -p test_speech_audio.py -v
python scripts/export_snippets.py --chapter ch24 --check
python -m examples.ch24_speech_audio.run_examples --output output/ch24.json
```

All ten tests pass and all three canonical listings match the manuscript.
The numerical helpers are unchanged. Raw squared FFT magnitudes are not a
calibrated power spectral density. Empty references have undefined individual
WER, but their insertions contribute to a corpus containing reference words.
