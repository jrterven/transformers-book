# Chapter 19: complete worked solutions

All [12 exercises](../../examples/ch19_generation/EXERCISES.md) have complete
explanations, intermediate calculations, and results in every edition.

- [Written solutions](written-solutions.pdf): every explanation, table, and plot.
- [Executed notebook](python-solutions.ipynb): the same solutions with 18 code cells.
- [Notebook PDF](python-solutions.pdf): all prose, code, plots, and saved results.

Start with facts, shifted targets, and complete path probabilities. Then
inspect clipped counts, an LCS table, and corpus aggregation. The search
investigations distinguish finished answers from prefixes stopped by a cap.
The final exercises separate source support, required-fact coverage, and
preprocessing failures. The truncation illustration uses word tokens; the
adaptation protocol is proposed, not an experiment claimed to have run.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch19_generation/python-solutions.ipynb`, restart its Python
kernel, and run all cells in order. No network, model, or dataset download
is needed. The saved run uses Python 3.12.14 and NumPy 2.5.4 on CPU.
Both plots are embedded in the notebook and retained in both PDF editions.

## Rebuild the PDFs

Install `mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch19_generation/build_pdfs.py
```

Use `--engine /path/to/tectonic` if necessary. The exporter verifies twelve
ordered solution headings, sequential cell execution, a recorded result for
each exercise, no notebook errors, and no significant typesetting warnings.
It builds both editions from this one notebook. Regenerate both after edits.

## Check the canonical examples

```sh
python -m unittest discover -s tests -p test_generation.py -v
python scripts/export_snippets.py --chapter ch19 --check
python -m examples.ch19_generation.run_examples --output output/ch19.json
```

The optional real encoder-decoder update requires PyTorch:

```sh
python -m pip install -r requirements-ch06.txt
python -m examples.ch19_generation.run_examples --train-step
```

All 11 chapter tests passed with that update included. Five manuscript
listings match the unchanged canonical helpers. The notebook additionally
checks corpus versus sentence aggregation and a causal claim with ROUGE-L
F1 of 0.9. Its support judgment comes from reading the source, not the metric.
