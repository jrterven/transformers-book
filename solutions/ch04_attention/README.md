# Chapter 4 | Worked solutions

All [15 chapter exercises](../../examples/ch04_attention/EXERCISES.md) have the
same numbering in the book, notebook, and PDFs.

- [Written solutions](written-solutions.pdf): every problem, explanation, equation, worked table, and interpretation.
- [Executed notebook](python-solutions.ipynb): the same solutions with runnable checks and recorded outputs.
- [Notebook PDF](python-solutions.pdf): explanations, code, and executed outputs together.

Exercises 1–8 develop the objects and hand calculations, 9–13 test axes and
information flow, and 14–15 are optional calculus challenges. The notebook is
the editable source for both PDFs and imports the canonical attention code.
The final projection-gradient exercise adds its own independent numerical check.

From the repository root:

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab solutions/ch04_attention/python-solutions.ipynb
```

Restart the kernel and run all cells in order. The recorded run executes all
25 code cells on a CPU with Python 3.12.14 and NumPy 2.5.4. It needs no training,
downloads, or accelerator. Small floating-point differences are possible.

To rebuild both PDFs, install `mistune` and Tectonic or XeLaTeX:

```sh
python -m pip install mistune
python solutions/ch04_attention/build_pdfs.py
```

Use `--engine /absolute/path/to/tectonic` if necessary. The exporter typesets
mathematics directly, checks solutions 1–15 in order, requires all code cells
to have run and each exercise to have a recorded result, and rejects execution
errors. Intermediate LaTeX and logs go under `output/ch04-solutions-build/`.
