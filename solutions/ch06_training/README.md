# Chapter 6 worked solutions

Juan Terven and Diana Córdova

All **15 exercises** in [EXERCISES.md](../../examples/ch06_training/EXERCISES.md)
have matching solutions in each edition:

- [Written solutions](written-solutions.pdf): explanations, calculations, and measured results for all exercises.
- [Executed notebook](python-solutions.ipynb): the same reasoning with runnable code and saved outputs.
- [Notebook PDF](python-solutions.pdf): explanations, code, and recorded outputs in order.

Exercises 1–7 isolate updates, data alignment, loss, clipping, and cache storage.
Exercises 8–11 test stopping, learning, checkpoint restoration, and length
extrapolation. Exercises 12–15 investigate model selection, information
leakage, cache errors, and timing. Every experiment uses locally generated data.
Temporary mask patches restore both imported functions; no source files change.

## Run

From the repository root:

```sh
python -m pip install -r requirements-ch06.txt jupyterlab
jupyter lab solutions/ch06_training/python-solutions.ipynb
```

Restart the kernel and run all cells. The notebook includes a 2,000-update
reference run plus smaller controlled experiments; allow a few minutes on a
CPU. Its numerical environment is printed in the first code cell. The saved
run uses Python 3.12.14, PyTorch 2.15.0, and one CPU thread. Float64 is used for
close numerical comparisons; training uses float32. Different versions can
change the training trajectory, selected checkpoint, and measured timings.

The reference experiment is also available independently:

```sh
python -m examples.ch06_training.run_experiment --output output/ch06
python -m unittest discover -s tests -p test_training.py -v
python scripts/export_snippets.py --chapter ch06 --check
```

The nine exported listings are generated from canonical source. The tests
cover loss normalization, data splits, stopping, fixed generation limits,
checkpoint recovery, learning, and cached/full-prefix agreement.

## Rebuild both PDFs

The executed notebook is the single source for both editions. Install
`mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch06_training/build_pdfs.py
```

Use `--engine /path/to/tectonic` or `--engine /path/to/xelatex` when needed.
The exporter checks ordered execution, complete exercise coverage, and
recorded results before typesetting. Inspect the rendered pages after edits.
