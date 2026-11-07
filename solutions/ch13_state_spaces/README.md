# Chapter 13: complete worked solutions

All [11 exercises](../../examples/ch13_state_spaces/EXERCISES.md) have complete
explanations, calculations, and results in each edition.

- [Written solutions](written-solutions.pdf): every derivation, table, and diagram.
- [Executed notebook](python-solutions.ipynb): all solutions, 17 code cells, and two plots.
- [Notebook PDF](python-solutions.pdf): every explanation, code cell, and saved output.

Follow a state and its lag filter, compare scan/direct/FFT calculations,
and diagnose circular wraparound, chunk resets, and gate placement. The
optional exercises account for saved history, distinguish carried state
from total memory, and separate linearity from time invariance.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch13_state_spaces/python-solutions.ipynb`, restart the
Python kernel, and run every cell in order. The saved CPU run uses Python
3.12.14 and NumPy 2.4.6. No model or dataset download is needed. The two
saved plots show future leakage from a circular convolution and the
effect of preserving or resetting a state at a chunk boundary.

## Export both PDF editions

Install `mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch13_state_spaces/build_pdfs.py
```

Use `--engine /path/to/tectonic` if needed. The exporter reads the executed
notebook, checks all eleven headings and recorded results, and generates
both PDFs. Temporary typesetting files go to `output/ch13-solutions-build/`.
Regenerate both editions after changing explanations or saved outputs.

## Validate the canonical examples

```sh
python -m unittest discover -s tests -p test_state_spaces.py -v
python scripts/export_snippets.py --chapter ch13 --check
python -m examples.ch13_state_spaces.run_examples --output output/ch13.json
```

All 11 tests passed. All five listings match the canonical source. These
NumPy helpers expose the arithmetic; they do not implement S4's fast
structured kernel construction or a full trained Hyena layer.
