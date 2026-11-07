# Chapter 10: complete worked solutions

All [12 exercises](../../examples/ch10_kv/EXERCISES.md) have complete
explanations, calculations, and results in each edition.

- [Written solutions](written-solutions.pdf): every derivation, table, and mask diagram.
- [Executed notebook](python-solutions.ipynb): the same solutions with 19 code cells and saved outputs.
- [Notebook PDF](python-solutions.pdf): every explanation, code cell, diagram, and recorded result.

Begin with four hand calculations, continue with four Python investigations,
and then try four optional extensions. Follow the timeline from sampling an
ID to storing its key and value, compare full and cached attention, check
future-value invariance, count actual cache bytes, and verify the two orders
of MLA computation. The last exercise proposes an inference measurement
protocol; it does not report a benchmark that was run.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch10_kv/python-solutions.ipynb`, select the installed
Python kernel, restart it, and run all cells in order. The saved run uses
Python 3.12.14 and NumPy 2.4.6 on CPU. No model or data download is needed.
Floating-point checks declare their tolerances. Masks, cache contents, and
integer byte counts are checked exactly where appropriate. The float32
experiment intentionally illustrates storage rounding.

## Export both PDF editions

Both PDFs are generated from the same executed notebook, including its
embedded mask diagram. Install `mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch10_kv/build_pdfs.py
```

Use `--engine /path/to/tectonic` if needed. Temporary typesetting files go
under `output/ch10-solutions-build/`. The exporter verifies all 12 solution
headings, sequential execution counts, results for Exercises 1-11, and the
absence of notebook errors or significant typesetting warnings. Exercise 12
is conceptual. Regenerate both PDFs after changing prose, code, or results.

## Check the canonical examples

```sh
python -m unittest discover -s tests -p test_kv.py -v
python -m unittest discover -s tests -p test_latent.py -v
python scripts/export_snippets.py --chapter ch10 --check
python -m examples.ch10_kv.run_examples --output output/ch10.json
```

The detailed review passed all 20 chapter tests and verified all five
manuscript listings against their canonical sources. Tests cover grouped
heads, per-sequence masks, chunk schedules, storage dtype, malformed inputs,
and expanded versus absorbed latent attention. The notebook adds the check
that changing a future value leaves earlier outputs unchanged.
