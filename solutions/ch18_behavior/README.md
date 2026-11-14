# Chapter 18: complete worked solutions

The [16 exercises](../../examples/ch18_behavior/EXERCISES.md) follow a
model's score from an intervention to an action choice. They include hand
calculations for attention writes, integrated gradients, sparse codes,
preference losses, and feedback optimization.

- [Written PDF](written-solutions.pdf)
- [Executed notebook](python-solutions.ipynb)
- [Notebook PDF](python-solutions.pdf)

The notebook uses the standard library and the recorded Qwen results. It
does not load the model or require any training. Open it in Jupyter with
the repository available, then restart the kernel and run all cells.

Rebuild the PDFs from the repository root with Mistune and Tectonic or
XeLaTeX installed.

```sh
python -m pip install mistune
python solutions/ch18_behavior/build_pdfs.py
```
