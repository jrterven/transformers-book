# Chapter 9: complete worked solutions

All [14 exercises](../../examples/ch09_scaling/EXERCISES.md) have complete
explanations, calculations, and results in each edition.

- [Written solutions](written-solutions.pdf): all derivations, tables, and the allocation plot.
- [Executed notebook](python-solutions.ipynb): the same solutions with 23 code cells and saved outputs.
- [Notebook PDF](python-solutions.pdf): every explanation, code cell, plot, and recorded result.

Start with six hand calculations, continue with five numerical investigations,
then try the three optional extensions. The examples distinguish corpus size
from token presentations, stored weights from per-token use, requests from
accepted assignments, and overflow from slot utilization. The gate comparison
shows why selected-logit top-one normalization differs from Switch routing.
The final exercise specifies a fair-comparison protocol; it does not report
an experiment that was run.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch09_scaling/python-solutions.ipynb`, select the installed
Python kernel, restart it, and run all cells in order. The saved run uses
Python 3.12.14 and NumPy 2.4.6 on CPU. It requires no model or data download.
Float64 checks declare their tolerances; integer IDs, masks, and counts are
checked exactly. Small last-digit differences can occur across versions.

## Export both PDF editions

Both PDFs are generated from the same executed notebook, including the
embedded allocation plot. Install `mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch09_scaling/build_pdfs.py
```

Use `--engine /path/to/tectonic` if needed. Temporary typesetting files go
under `output/ch09-solutions-build/`. The exporter verifies all 14 solution
headings, sequential execution counts, results for Exercises 1-13, and the
absence of notebook errors or significant typesetting warnings. Exercise 14
is conceptual. Regenerate both PDFs whenever prose, code, or results change.

## Check the canonical examples

```sh
python -m unittest discover -s tests -p test_scaling.py -v
python scripts/export_snippets.py --chapter ch09 --check
python -m examples.ch09_scaling.run_examples --output output/ch09.json
```

The detailed review passed all 16 chapter tests and verified all four
manuscript listings against their canonical sources. Regression checks cover
the two-expert update, selected-gate normalization, tie handling, and the
counterexample showing that a balance term of 1 need not mean balanced requests.
