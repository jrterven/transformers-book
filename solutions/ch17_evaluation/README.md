# Chapter 17: complete worked solutions

All [twelve exercises](../../examples/ch17_evaluation/EXERCISES.md) are solved
in every edition:

- [Written PDF](written-solutions.pdf): all calculations, interpretations and designs.
- [Executed notebook](python-solutions.ipynb): the same explanations, 21 code cells and two plots.
- [Notebook PDF](python-solutions.pdf): all code and saved outputs.

The sequence covers evaluation units, error costs, reliability and Brier
scores, paired outcomes, cost-based thresholds, group splits, empty bins,
bootstrap units, prediction joins, long-context controls, ablations and
reproducibility. These are synthetic teaching fixtures, not model benchmarks.

## Run and export

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab mistune
jupyter lab
```

Open the notebook, restart the kernel and run all cells. The saved run uses
Python 3.12.14, NumPy 2.5.4 and Matplotlib 3.11.2 on CPU. No external models
or datasets are needed. Seeds are explicit in the split and bootstrap calls.

With Tectonic or XeLaTeX installed, rebuild both PDF editions:

```sh
python solutions/ch17_evaluation/build_pdfs.py
```

Use `--engine /path/to/tectonic` if needed. The exporter requires twelve
ordered exercise headings, sequential execution counts, and a saved result
for every exercise. Temporary files go to `output/ch17-solutions-build/`.
Rebuild both PDFs when an explanation, calculation or plot changes.

## Canonical validation

```sh
python -m unittest discover -s tests -p test_evaluation.py -v
python -m examples.ch17_evaluation.run_examples --output output/ch17.json
python scripts/export_snippets.py --check
```

All eleven chapter tests pass, and all four manuscript excerpts match the
canonical source. The notebook also checks the three scoring units, a cost
threshold, an unequal-customer weighting reversal, invalid joins and explicit
evidence-to-question distances. Its plots show empty-bin handling and the
two resampling intervals. The canonical helpers remain unchanged.
