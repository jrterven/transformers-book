# Chapter 21: complete worked solutions

All [12 exercises](../../examples/ch21_retrieval/EXERCISES.md) have complete
explanations and intermediate calculations in each edition.

- [Written solutions](written-solutions.pdf): explanations, tables and both plots.
- [Executed notebook](python-solutions.ipynb): 17 executed cells with saved outputs.
- [Notebook PDF](python-solutions.pdf): all prose, code, figures and results.

The sequence progresses from BM25, vector scores, context budgets and recall
to RAG probability mixtures, code investigations and source audits. Two plots
show the length adjustment and evidence loss during packing. Design exercises
include controlled annotations and evaluation protocols, not model benchmarks.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch21_retrieval/python-solutions.ipynb`, restart its Python
kernel and run all cells in order on CPU. No model, dataset or network is
needed. The saved run uses Python 3.12.14, NumPy 2.5.4 and Matplotlib 3.11.2.

## Rebuild both PDFs

Install `mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch21_retrieval/build_pdfs.py
```

Use `--engine /path/to/tectonic` if necessary. The exporter checks all twelve
solution headings, sequential cell execution, a recorded result per exercise,
absence of notebook errors and significant typesetting warnings. It builds
both PDFs from the same notebook; regenerate both after changes.

## Verify the canonical examples

```sh
python -m unittest discover -s tests -p test_retrieval.py -v
python scripts/export_snippets.py --chapter ch21 --check
python -m examples.ch21_retrieval.run_examples --output output/ch21.json
```

All nine chapter tests passed and five canonical snippets match the manuscript.
The helper functions remain unchanged. A raw retrieval score is not a
probability of answer support, a phrase occurrence is not entailment, and
an empty annotation set alone does not establish unanswerability.
