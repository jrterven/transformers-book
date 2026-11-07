# Chapter 18: complete worked solutions

All [twelve exercises](../../examples/ch18_understanding/EXERCISES.md) are solved
in every edition:

- [Written PDF](written-solutions.pdf): all calculations, interpretations and study designs.
- [Executed notebook](python-solutions.ipynb): the same explanations, 19 code cells and two plots.
- [Notebook PDF](python-solutions.pdf): all code and saved outputs.

The sequence covers word order, masked pooling, IOB2 labels, exact entity
counts, sentence identity and micro aggregation, padding, first-piece alignment,
repair, boundary/type errors, prevalence, independent splits and error studies.
The supplied arrays isolate heads and metrics; they do not benchmark a trained encoder.

## Run and export

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab mistune
jupyter lab
```

Open the notebook, restart the kernel and run all cells. The saved run uses
Python 3.12.14, NumPy 2.5.4 and Matplotlib 3.11.2 on CPU. It needs no external
models or datasets. Both plots are saved in the notebook.

With Tectonic or XeLaTeX installed, rebuild both PDF editions:

```sh
python solutions/ch18_understanding/build_pdfs.py
```

Use `--engine /path/to/tectonic` if needed. The exporter requires twelve ordered
exercise headings, sequential execution counts and a result for every exercise.
Temporary files go to `output/ch18-solutions-build/`. Rebuild both PDFs when an
explanation, calculation or plot changes.

## Canonical validation

```sh
python -m unittest discover -s tests -p test_understanding.py -v
python -m examples.ch18_understanding.run_examples --output output/ch18.json
python scripts/export_snippets.py --check
```

All nine chapter tests pass; all four manuscript excerpts match their canonical
source. The notebook additionally checks cross-sentence identity, micro versus
mean-sentence F1, changing class prevalence and the assignment of incomplete
predicted names to the full gold name's slice. The canonical helpers are unchanged.
