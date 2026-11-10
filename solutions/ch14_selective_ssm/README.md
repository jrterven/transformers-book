# Chapter 14: complete worked solutions

All [10 exercises](../../examples/ch14_selective_ssm/EXERCISES.md) have complete
explanations, calculations, and results in each edition.

- [Written solutions](written-solutions.pdf): all derivations, tables, and plots.
- [Executed notebook](python-solutions.ipynb): all solutions, 21 code cells, and two plots.
- [Notebook PDF](python-solutions.pdf): every explanation, code cell, and saved output.

Trace retain and write contributions, connect a step size to its sigmoid gate,
compose updates without reordering them, and follow individual input-to-output
paths. Compare dense, recurrent, and chunked SSD, including short final chunks,
causality, and an exact discrete reset. The memory-test design includes a
two-slot first-and-latest oracle and a query it cannot answer from those slots.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch14_selective_ssm/python-solutions.ipynb`, restart the Python
kernel, and run all cells. The saved CPU run uses Python 3.12.14 and NumPy
2.4.6. The plots compare two coordinate retentions and show how perturbing
one input affects only current and later outputs, even across chunk boundaries.
No model or dataset download is required. The memory counts are illustrative
array payloads, not learned-model accuracy or runtime measurements.

## Export both PDF editions

Install `mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch14_selective_ssm/build_pdfs.py
```

Use `--engine /path/to/tectonic` if needed. The exporter reads the executed
notebook and requires all ten exercise headings and their recorded results.
Temporary typesetting files go to `output/ch14-solutions-build/`.
Regenerate both PDF editions after changing explanations or outputs.

## Validate the canonical examples

```sh
python -m unittest discover -s tests -p test_selective_ssm.py -v
python scripts/export_snippets.py --chapter ch14 --check
python -m examples.ch14_selective_ssm.run_examples --output output/ch14.json
```

All 16 tests passed; all six exported listings match the canonical code.
The helpers use precomputed coefficients for simplified SSM cores. They do
not implement a full Mamba block, a trained control network, or a GPU scan.
