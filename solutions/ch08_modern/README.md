# Chapter 8: complete worked solutions

All [14 exercise statements](../../examples/ch08_modern/EXERCISES.md) match
the revised chapter. Start with six hand calculations, continue with five
small numerical investigations, then try the three optional extensions.

- [Written solutions](written-solutions.pdf): every explanation, derivation, and result.
- [Executed notebook](python-solutions.ipynb): the same explanations with 20 code cells and saved outputs.
- [Notebook PDF](python-solutions.pdf): the complete notebook typeset for reading and printing.

The examples explain the residual route, RMSNorm geometry and epsilon,
SwiGLU gates and parameter budgets, RoPE signs and multiple frequencies,
causal and padding masks, common position offsets, cache indices, and
normalization gradients. Exercise 13 compares two losses and checks a nonzero
post-norm gradient against central finite differences. Exercise 14 specifies
a controlled experiment; it does not claim an observed winner.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements-ch05.txt jupyterlab
jupyter lab
```

Open `solutions/ch08_modern/python-solutions.ipynb`, select this environment's
Python kernel, restart it, and run all cells in order. The saved run uses
Python 3.12.14, PyTorch 2.15.0, and NumPy 2.4.6 on one CPU thread. All numerical
probes use float64 and declare their tolerances. No model or corpus download
is needed. Small last-digit differences can occur across software versions.

## Export both PDF editions

Both PDFs come from the same executed notebook. Install `mistune` and either
Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch08_modern/build_pdfs.py
```

If needed, supply `--engine /path/to/tectonic`. Temporary typesetting files
go under `output/ch08-solutions-build/`. The exporter checks the 14 headings,
sequential execution counts, saved results for exercises 1-13, absence of
notebook errors, and significant typesetting warnings. Regenerate both PDFs
whenever explanations, code, or outputs change.

## Check the canonical examples

```sh
python -m unittest discover -s tests -p test_modern.py -v
python scripts/export_snippets.py --chapter ch08 --check
python -m examples.ch08_modern.run_examples --output output/ch08.json
```

The November 6, 2026 review passed all 16 chapter tests without skips and
verified all seven exported listings. The two new regression checks cover
the sum of two rotary pair scores and the difference between a common
position shift and an added prefix token.
