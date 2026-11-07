# Chapter 12: complete worked solutions

All [11 exercises](../../examples/ch12_efficient/EXERCISES.md) have complete
explanations, calculations, and results in each edition.

- [Written solutions](written-solutions.pdf): every derivation, table, and diagram.
- [Executed notebook](python-solutions.ipynb): all solutions, 16 code cells, and two plots.
- [Notebook PDF](python-solutions.pdf): every explanation, code cell, and saved output.

Begin with response timing and byte counts. Trace the online softmax state in
both tile orders, then study partial tiles, empty rows, absolute positions,
and large-score stability. The final exercises distinguish a local kernel
speedup from an end-to-end improvement and propose a fair measurement protocol.
The proposed GPU protocol is not a benchmark that was run.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch12_efficient/python-solutions.ipynb`, restart the Python
kernel, and run all cells in order. The saved run uses Python 3.12.14 and
NumPy 2.4.6 on CPU. No model or data download is needed. Exact counts and
masks use exact checks; floating-point results use explicit tolerances.
The two saved plots show the uneven score tiles and the causal-offset masks.

## Export both PDF editions

Install `mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch12_efficient/build_pdfs.py
```

Use `--engine /path/to/tectonic` if needed. The exporter reads the executed
notebook, checks all eleven headings and the recorded results for Exercises
1-10, and writes both PDFs. Exercise 11 is conceptual. Temporary typesetting
files go to `output/ch12-solutions-build/`. Regenerate both editions after
changing explanations, calculations, or saved outputs.

## Validate the canonical code

```sh
python -m unittest discover -s tests -p test_efficient.py -v
python scripts/export_snippets.py --chapter ch12 --check
python -m examples.ch12_efficient.run_examples --output output/ch12.json
```

All nine tests passed. All three manuscript listings match their canonical
sources. The code checks the exact attention formula using float64 NumPy;
CPU timings for this teaching implementation do not measure a fused GPU kernel.
