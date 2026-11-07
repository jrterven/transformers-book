# Chapter 28: complete worked solutions

All [11 exercises](../../examples/ch28_tabular/EXERCISES.md) are solved
step by step in each edition:

- [Written solutions](written-solutions.pdf): explanations, calculations, tables and three plots.
- [Executed notebook](python-solutions.ipynb): all explanations with 20 executed code cells.
- [Notebook PDF](python-solutions.pdf): prose, code, figures and saved results.

The plots explain expected Brier loss, the influence of a negative context row,
and calibration versus risk separation. The notebook preserves the detailed
row-encoding and probability calculations. Evaluation designs include label
maturity, grouped and chronological splits, equal task-label budgets, and
separate calibration periods. Synthetic fixtures are not model benchmarks.

## Run and reproduce

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch28_tabular/python-solutions.ipynb`, restart the kernel
and run all cells. The saved run used Python 3.12.14, NumPy 2.5.4 and
Matplotlib 3.11.2. No model or dataset downloads are needed.

To export both PDFs from the executed notebook, install Tectonic or XeLaTeX:

```sh
python -m pip install mistune
python solutions/ch28_tabular/build_pdfs.py
```

An explicit engine path can be given with `--engine`. The exporter checks
all eleven headings, ordered execution, recorded exercise results and
absence of execution errors.

## Canonical checks

```sh
python -m unittest discover -s tests -p test_tabular.py -v
python scripts/export_snippets.py --chapter ch28 --check
python -m examples.ch28_tabular.run_examples --output output/ch28.json
```

All eight tests pass and all four canonical listings match the manuscript.
The tabular helpers are unchanged.
