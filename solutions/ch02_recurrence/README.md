# Chapter 2 | Worked solutions

These materials cover all [14 chapter exercises](../../examples/ch02_recurrence/EXERCISES.md), with the same numbering everywhere.

- [Written solutions](written-solutions.pdf): explanations, equations, worked tables, and interpretation of the training results.
- [Executed notebook](python-solutions.ipynb): the same solutions with runnable checks and recorded outputs.
- [Notebook PDF](python-solutions.pdf): explanations, code, and the executed results in one typeset document.

The notebook is the editable source for both PDFs. It imports the canonical Chapter 2 implementation; it does not keep a second model implementation.

From the repository root, install the example and notebook dependencies:

```sh
python -m pip install -r requirements-ch02.txt jupyterlab
jupyter lab solutions/ch02_recurrence/python-solutions.ipynb
```

Restart the kernel and run all cells in order. Exercise 12 trains for 750 updates on one CPU thread in a temporary directory. It reloads the checkpoint selected by validation loss; Exercise 13 uses that same run. The recorded reference run uses Python 3.12.14, PyTorch 2.15.0, and NumPy 2.5.4. Small numerical differences across environments are possible.

To rebuild the written and notebook PDFs, install `mistune` and either Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch02_recurrence/build_pdfs.py
```

Use `--engine /absolute/path/to/tectonic` if the compiler is not on PATH. The exporter typesets mathematical notation directly, requires exercises 1-14 and complete executed outputs, and rejects missing results or execution errors. Intermediate LaTeX files and compilation logs go under `output/ch02-solutions-build/` by default.

The extrapolation split proposed in Exercise 13 is an experimental design, not a measured result. The actual run evaluates new color arrangements at lengths present in training.
