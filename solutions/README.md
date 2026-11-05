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

## Chapter 3. Learning to Align

| Exercises | Topics | Files |
| --- | --- | --- |
| 1, 3, 4, 6, and 7 | Weighted reads, normalization axes, source masks, query dependence, and tensor shapes | [Written solutions (PDF)](ch03_alignment/written-solutions.pdf) |
| 2, 5, and 8–12 | Stable softmax, additive scores, source padding, decoder timing, learned attention maps, and annotation order | [Jupyter notebook](ch03_alignment/python-solutions.ipynb) · [Notebook export (PDF)](ch03_alignment/python-solutions.pdf) |

To rerun the notebook, start JupyterLab from the repository root in an
environment with the [Chapter 3 dependencies](../requirements-ch03.txt).

```sh
python -m pip install -r requirements-ch03.txt jupyterlab
jupyter lab
```

Open `solutions/ch03_alignment/python-solutions.ipynb`, select that Python
kernel, and run all cells. The notebook runs the 750-update CPU experiment
from the [Chapter 3 examples](../examples/ch03_alignment/) and displays the
attention weights along actual prediction paths.

## Chapter 4. Self-Attention from Vectors to Matrices

| Exercises | Topics | Files |
| --- | --- | --- |
| 1, 2, and 4–6 | Uniform weights, changed values, sequence shapes, asymmetric attention, and causal normalization | [Written solutions (PDF)](ch04_attention/written-solutions.pdf) |
| 3 and 7–12 | One query, batch and head axes, masks, permutations, finite differences, and gradients through projections | [Jupyter notebook](ch04_attention/python-solutions.ipynb) · [Notebook export (PDF)](ch04_attention/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with NumPy.

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch04_attention/python-solutions.ipynb`, select that Python
kernel, and run all cells. The notebook uses the NumPy implementation from the
[Chapter 4 examples](../examples/ch04_attention/).

## Chapter 5. Multiple Heads, Position, and the Complete Architecture

| Exercises | Topics | Files |
| --- | --- | --- |
| 2 and 3 | Attention map shapes and sinusoidal positions | [Written solutions (PDF)](ch05_transformer/written-solutions.pdf) |
| 1 and 4–12 | Heads, normalization, dropout, feed-forward layers, vocabulary loss, padding, causality, and deeper stacks | [Jupyter notebook](ch05_transformer/python-solutions.ipynb) · [Notebook export (PDF)](ch05_transformer/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[Chapter 5 dependencies](../requirements-ch05.txt).

```sh
python -m pip install -r requirements-ch05.txt jupyterlab
jupyter lab
```

Open `solutions/ch05_transformer/python-solutions.ipynb`, select that Python
kernel, and run all cells. The notebook uses the CPU PyTorch implementation
from the [Chapter 5 examples](../examples/ch05_transformer/).

## Chapter 6. Training and Generating with a Small Transformer

| Exercises | Topics | Files |
| --- | --- | --- |
| 2, 4, and 5 | Token loss and its gradient, gradient clipping, and KV cache storage | [Written solutions (PDF)](ch06_training/written-solutions.pdf) |
| 1, 3, and 6–12 | Padded batches, corpus NLL, generation, model selection, leakage, cache correctness, and timing | [Jupyter notebook](ch06_training/python-solutions.ipynb) · [Notebook export (PDF)](ch06_training/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[Chapter 6 dependencies](../requirements-ch06.txt).

```sh
python -m pip install -r requirements-ch06.txt jupyterlab
jupyter lab
```

Open `solutions/ch06_training/python-solutions.ipynb`, select that Python
kernel, and run all cells. The notebook uses CPU models and the synthetic
color sequences from the [Chapter 6 examples](../examples/ch06_training/).
It includes the reference training recipe and four optional experiments.
