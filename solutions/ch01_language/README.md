# Chapter 1 worked solutions

The [12 exercise statements](../../examples/ch01_language/EXERCISES.md) match the
revised manuscript. The numbering progresses from representations and shapes
to probability calculations, coding, and two challenges.

- [Written solutions](written-solutions.pdf): explanations and answers for all 12 exercises.
- [Executed notebook](python-solutions.ipynb): the same explanations plus runnable checks for every exercise.
- [Notebook PDF](python-solutions.pdf): the full notebook, including code and recorded outputs.

Run the notebook from this folder or the repository root in an environment with
NumPy installed. It locates the canonical functions in
`examples/ch01_language/model.py`. The examples use fixed teaching weights and
require no training, GPU, dataset, or model download.

To execute and rebuild the PDFs from the repository root:

```sh
python -m pip install -r requirements.txt nbconvert ipykernel reportlab mistune
python -m nbconvert --execute --to notebook --inplace \
  solutions/ch01_language/python-solutions.ipynb
python solutions/ch01_language/build_pdfs.py
```

The notebook is the editable solution source. The written PDF exports its
explanatory Markdown; the notebook PDF additionally exports every code cell
and its output. The exporter refuses unexecuted or failed notebooks and
out-of-order execution counts. Exercise 12 deliberately displays an infinite
loss from `-log(0)` alongside the finite result computed directly from logits.
That displayed infinity is the expected numerical demonstration.
