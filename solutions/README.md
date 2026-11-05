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

## Chapter 2. Recurrence and Encoder-Decoder Models

| Exercises | Topics | Files |
| --- | --- | --- |
| 2–8 and 12 | Recurrent order, gradient paths, LSTM gates, encoder state transfer, shifted targets, batch shapes, loss, and the information path | [Written solutions (PDF)](ch02_recurrence/written-solutions.pdf) |
| 1 and 9–11 | The scalar state trace, packed source padding, teacher-forced scoring, greedy generation, and evaluation by source length | [Jupyter notebook](ch02_recurrence/python-solutions.ipynb) · [Notebook export (PDF)](ch02_recurrence/python-solutions.pdf) |

To rerun the notebook, start JupyterLab from the repository root in an
environment with the [Chapter 2 dependencies](../requirements-ch02.txt).

```sh
python -m pip install -r requirements-ch02.txt jupyterlab
jupyter lab
```

Open `solutions/ch02_recurrence/python-solutions.ipynb`, select that Python
kernel, and run all cells. The notebook runs the 750-update CPU color-reversal
experiment from the [Chapter 2 examples](../examples/ch02_recurrence/).
