# Chapter 15: complete worked solutions

Every one of the [ten exercises](../../examples/ch15_hybrids/EXERCISES.md) is
fully solved in all three editions.

- [Written PDF](written-solutions.pdf): derivations, numerical results, tables and plots.
- [Executed notebook](python-solutions.ipynb): all explanations, 21 code cells and two plots.
- [Notebook PDF](python-solutions.pdf): every code cell and its saved results.

The sequence covers prediction and correction, signed carry coefficients,
Jamba's two schedules, mixed storage precisions, unit-key overwrites,
per-layer histories, causal cache equivalence, interfering writes, and a
controlled evaluation design. The circular-move task is explicitly separated
from harder, order-sensitive pointer swaps. No trained-model advantage is
inferred from the teaching examples.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch15_hybrids/python-solutions.ipynb`, restart its Python kernel,
and run all cells. The saved CPU run uses Python 3.12.14 and NumPy 2.5.4.
Plots are saved inline. No model weights or datasets need to be downloaded.
The figures show the fixed and growing parts of the toy stores and the
effect of correcting two overlapping associations in succession.

## Rebuild both PDFs

Install `mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch15_hybrids/build_pdfs.py
```

Use `--engine /path/to/tectonic` to specify the engine. The exporter checks
that all ten exercise headings and recorded results are present and that
all code cells were executed in order. Rebuild both PDFs after changes.
Temporary files go to `output/ch15-solutions-build/`.

## Check the canonical examples

```sh
python -m unittest discover -s tests -p test_hybrids.py -v
python scripts/export_snippets.py --chapter ch15 --check
python -m examples.ch15_hybrids.run_examples --output output/ch15.json
```

All nine tests pass and the four exported listings match their canonical
functions. The NumPy helpers isolate mixer updates and storage; they do not
reproduce complete Jamba or Olmo Hybrid models or optimized inference kernels.
