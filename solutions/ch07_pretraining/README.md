# Chapter 7: complete worked solutions

All [14 exercise statements](../../examples/ch07_pretraining/EXERCISES.md)
match the revised chapter. Begin with the seven short conceptual exercises,
continue with four small experiments, and then try the three optional
investigations. The final exercise designs a common evaluation protocol.

- [Written solutions](written-solutions.pdf): all explanations, calculations, and measured result tables.
- [Executed notebook](python-solutions.ipynb): the same explanations with 23 code cells and saved outputs.
- [Notebook PDF](python-solutions.pdf): the complete notebook typeset for reading and printing.

The examples run on one CPU thread with local synthetic data; no model or
corpus download is needed. The networks start from random weights. Fitting
one prepared example checks the computation, and the classification task
compares trainable components. Neither experiment measures pretrained transfer
or general language ability.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements-ch05.txt jupyterlab
jupyter lab
```

Open `solutions/ch07_pretraining/python-solutions.ipynb`, select this environment's
Python kernel, restart it, and run all cells in order. The saved run used
Python 3.12.14, PyTorch 2.15.0, and NumPy 2.4.6 on CPU. Seeds and exact data
are in the notebook. Numerical versions can change training trajectories and
selected checkpoints; the stated information-path invariants should still hold
within the declared tolerances.

`bert_corrupt(..., return_trace=True)` exposes the canonical helper's actual
decisions without consuming additional random draws. `encode_hidden(ids)`
returns position representations before the vocabulary head. These are the
same APIs used by the chapter's exported listings.

## Export the two PDF editions

Both PDFs are generated from the same executed notebook. Install `mistune`
and either Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch07_pretraining/build_pdfs.py
```

If the engine is not on PATH, pass `--engine /path/to/tectonic`. Temporary LaTeX
files are written under `output/ch07-solutions-build/`. The exporter requires
all 14 solution headings in order, sequential execution counts, and recorded
results for exercises 1-13. Exercise 14 is a conceptual protocol. It rejects
notebook errors and significant typesetting warnings. Regenerate both PDFs
whenever the notebook explanations, code, or outputs change.

## Check the canonical examples

```sh
python -m unittest discover -s tests -p test_pretraining.py -v
python scripts/export_snippets.py --chapter ch07 --check
python -m examples.ch07_pretraining.run_examples --train --steps 200 --output output/ch07.json
```

The November 6, 2026 review passed all 16 chapter tests without skips and
verified all seven snippets. The tests include causal and padding invariance,
selected-loss gradients, encoder-decoder paths, an unchanged random stream
when tracing corruption, and the hidden-vector API's shape and head gradients.
