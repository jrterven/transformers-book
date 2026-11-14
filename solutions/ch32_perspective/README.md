# Chapter 32: complete worked solutions

All [12 exercises](../../examples/ch32_perspective/EXERCISES.md) appear in:

- [Written solutions](written-solutions.pdf): explanations, calculations, tables and three figures.
- [Executed notebook](python-solutions.ipynb): all solutions with 19 executed code cells.
- [Notebook PDF](python-solutions.pdf): prose, code, figures and saved outputs.

The figures compare causal-pair growth, persistent-history budgets, and
exact-answer outcomes for four explicit memory rules. Counts and rule outputs
are teaching examples, not GPU timings or trained-model benchmarks.
The evaluation designs include matched address/activation changes and an
as-of weather-availability check.

## Run and reproduce

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch32_perspective/python-solutions.ipynb` and run all cells in order.
The saved run used Python 3.12.14, NumPy 2.5.4 and Matplotlib 3.11.2.
No model, external dataset or accelerator is required.

Install Tectonic or XeLaTeX and export both PDFs from the executed notebook:

```sh
python -m pip install mistune
python solutions/ch32_perspective/build_pdfs.py
```

Use `--engine` for an explicit executable path. The exporter requires twelve
ordered solution headings, sequential execution and a result for every exercise.

## Canonical checks

```sh
python -m unittest discover -s tests -p test_perspective.py -v
python scripts/export_snippets.py --chapter ch32 --check
python -m examples.ch32_perspective.run_examples --output output/ch32.json
```

Ten tests pass and two canonical snippets match the manuscript.

## Primary background

- [RULER: configurable long-context evaluations](https://arxiv.org/abs/2404.06654)
- [DeepSeek-V3.2: main attention and indexer costs](https://arxiv.org/abs/2512.02556)
- [Olmo Hybrid](https://arxiv.org/abs/2604.03444)
- [DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348)
