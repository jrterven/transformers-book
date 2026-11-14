# Chapter 28: complete worked solutions

All [11 exercises](../../examples/ch28_time_series/EXERCISES.md) are solved
step by step in each edition:

- [Written solutions](written-solutions.pdf): explanations, calculations, tables and three plots.
- [Executed notebook](python-solutions.ipynb): all explanations with 18 executed code cells.
- [Notebook PDF](python-solutions.pdf): prose, code, figures and saved results.

The plots show asymmetric quantile loss, seasonal drift increments and the
coverage-width tradeoff. Forecasting protocols are specified with exact time
boundaries. Synthetic examples illustrate the rules; they are not model benchmarks.

## Run and reproduce

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch28_time_series/python-solutions.ipynb`, restart the kernel
and run all cells. The saved run used Python 3.12.14, NumPy 2.5.4 and
Matplotlib 3.11.2. No model or dataset downloads are needed.

To export both PDFs from the executed notebook, install Tectonic or XeLaTeX:

```sh
python -m pip install mistune
python solutions/ch28_time_series/build_pdfs.py
```

An explicit engine path can be given with `--engine`. The exporter checks
all eleven headings, ordered execution, recorded exercise results and
absence of execution errors.

## Canonical checks

```sh
python -m unittest discover -s tests -p test_time_series.py -v
python scripts/export_snippets.py --chapter ch28 --check
python -m examples.ch28_time_series.run_examples --output output/ch28.json
```

All ten tests pass and all four canonical listings match the manuscript.
The forecasting helpers are unchanged.
