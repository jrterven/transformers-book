# Exercise solutions

## Chapter 1. Tokens, Embeddings, and Neural Language Models

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–7 and 9 | Token IDs, embedding lookup, vocabulary renumbering, subwords, softmax, context order, batch shapes, and sentence probabilities | [Written solutions (PDF)](ch01_language/written-solutions.pdf) |
| 8 and 10 | Next-token training pairs and the limits of a two-token context | [Jupyter notebook](ch01_language/python-solutions.ipynb) · [Notebook export (PDF)](ch01_language/python-solutions.pdf) |

The notebook includes its executed outputs. To rerun it, start JupyterLab from
the repository root in a Python environment with NumPy installed.

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch01_language/python-solutions.ipynb`, select the Python kernel
from that environment, and run all cells. The notebook uses the fixed weights
and functions from the [Chapter 1 examples](../examples/ch01_language/).
