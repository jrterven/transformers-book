# Chapter 24: complete worked solutions

All [11 exercises](../../examples/ch24_multimodal/EXERCISES.md) are explained
step by step in each edition:

- [Written solutions](written-solutions.pdf): calculations, explanations, tables and three plots.
- [Executed notebook](python-solutions.ipynb): all explanations and 14 executed cells.
- [Notebook PDF](python-solutions.pdf): all prose, code, figures and saved results.

The plots compare row and column probabilities, recall and any-hit scores,
and a paired bus/shelter spatial test. The final examples audit scene splits,
pair accuracy and separate judgments of ambiguity and unsupported claims.
These synthetic examples do not measure a pretrained visual assistant.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch24_multimodal/python-solutions.ipynb`, restart its kernel
and run all cells in order. The saved run uses Python 3.12.14, NumPy 2.5.4
and Matplotlib 3.11.2 on CPU. No model or dataset download is needed.

## Rebuild the PDFs

Install Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch24_multimodal/build_pdfs.py
```

Use `--engine /path/to/tectonic` if needed. The exporter verifies all eleven
solution headings, sequential execution, a result for each exercise and no
execution errors. Both PDFs come from the same executed notebook.

## Verify canonical examples

```sh
python -m unittest discover -s tests -p test_multimodal.py -v
python scripts/export_snippets.py --chapter ch24 --check
python -m examples.ch24_multimodal.run_examples --output output/ch24.json
```

Nine tests pass and four canonical listings match the manuscript. The
helpers are unchanged. Rectangular retrieval supports multiple positives;
its per-query recall differs from any-hit rate. A positive logit scale
changes probabilities without changing fixed-vector rankings.
