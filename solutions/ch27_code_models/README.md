# Chapter 27: complete worked solutions

All [11 exercises](../../examples/ch27_code_models/EXERCISES.md) are solved
step by step in each edition:

- [Written solutions](written-solutions.pdf): explanations, calculations, tables and three plots.
- [Executed notebook](python-solutions.ipynb): all explanations with 17 executed code cells.
- [Notebook PDF](python-solutions.pdf): prose, code, figures and saved results.

The plots show successful pairs, each candidate's test outcomes and correct
task-level aggregation. The repository experiments use known synthetic
contracts. They are evaluation designs, not model benchmark results.

## Run and reproduce

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch27_code_models/python-solutions.ipynb`, restart the kernel
and run all cells. The saved run used Python 3.12.14 and Matplotlib 3.11.2.
The canonical examples use the Python standard library. No models or datasets
are downloaded and no newly generated programs are executed.

To export both PDFs from the executed notebook, install Tectonic or XeLaTeX:

```sh
python -m pip install mistune
python solutions/ch27_code_models/build_pdfs.py
```

An explicit engine path can be given with `--engine`. The exporter checks
all eleven headings, ordered execution, recorded exercise results and
absence of execution errors.

## Canonical checks

```sh
python -m unittest discover -s tests -p test_code_models.py -v
python scripts/export_snippets.py --chapter ch27 --check
python -m examples.ch27_code_models.run_examples --output output/ch27.json
```

All seven tests pass and all four canonical listings match the manuscript.
The helpers are unchanged. The scorer calls known local functions and requires
actual Python Booleans; it is not an execution sandbox.
