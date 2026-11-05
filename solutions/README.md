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

## Chapter 7. GPT, BERT, and T5

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–5 and 12 | Shifted targets, attention grids, BERT selection and replacement, T5 span reconstruction, uniform loss, and fair comparison | [Written solutions (PDF)](ch07_pretraining/written-solutions.pdf) |
| 6–11 | BERT replacement traces, loss gradients, causal and encoder–decoder information paths, a training run, and a classification head | [Jupyter notebook](ch07_pretraining/python-solutions.ipynb) · [Notebook export (PDF)](ch07_pretraining/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[PyTorch dependencies](../requirements-ch05.txt).

```sh
python -m pip install -r requirements-ch05.txt jupyterlab
jupyter lab
```

Open `solutions/ch07_pretraining/python-solutions.ipynb`, select that Python
kernel, and run all cells. The notebook uses the
[Chapter 7 teaching models](../examples/ch07_pretraining/), runs on CPU,
and needs no datasets or model weights from the internet. The notebook PDF
includes the code and its recorded outputs. All three optional exercises
are covered across the two PDFs.

## Chapter 8. From the Original Block to Modern Decoders

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–5 and 12 | Normalization placement and scaling, SwiGLU calculations and parameter budgets, the sign in RoPE, and a matched experiment | [Written solutions (PDF)](ch08_modern/written-solutions.pdf) |
| 6–11 | RMSNorm with epsilon, loading gate weights, causal and padding masks, decoder parameter counts, rotary positions in a cache, and residual gradients | [Jupyter notebook](ch08_modern/python-solutions.ipynb) · [Notebook export (PDF)](ch08_modern/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[PyTorch dependencies](../requirements-ch05.txt).

```sh
python -m pip install -r requirements-ch05.txt jupyterlab
jupyter lab
```

Open `solutions/ch08_modern/python-solutions.ipynb`, select that Python
kernel, and run all cells. The notebook uses the
[Chapter 8 teaching model](../examples/ch08_modern/) and runs on CPU with
no external datasets or pretrained weights. The PDF export includes the code
and its recorded outputs. The three optional exercises are covered across
the two PDFs.

## Chapter 9. Scaling Data, Compute, and Expert Capacity

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4, 10, and 12 | Compute budgets, token presentations, expert gates, stored and used weights, device traffic, and a dense versus routed comparison | [Written solutions (PDF)](ch09_scaling/written-solutions.pdf) |
| 5–9 and 11 | Allocation sweeps, stable softmax, capacity and overflow, residual bypass, router balance, and the continuous optimum | [Jupyter notebook](ch09_scaling/python-solutions.ipynb) · [Notebook export (PDF)](ch09_scaling/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[NumPy requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch09_scaling/python-solutions.ipynb`, select that Python
kernel, and run all cells. It uses the
[Chapter 9 examples](../examples/ch09_scaling/) on CPU with fixed inputs.
The PDF export includes the code and its recorded outputs. The two PDFs
cover all three optional exercises.

## Chapter 10. Incremental Decoding and KV Representations

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–12 | Prefill and decoding, attention work, shared heads, cache budgets, rotary positions, latent score scaling, and inference measurement | [Written solutions (PDF)](ch10_kv/written-solutions.pdf) |
| 5–8 | Chunk masks, cached attention equality, storage dtype and byte counts, and expanded versus latent attention | [Jupyter notebook](ch10_kv/python-solutions.ipynb) · [Notebook export (PDF)](ch10_kv/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[NumPy requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch10_kv/python-solutions.ipynb`, select that Python kernel,
and run all cells. It uses the [Chapter 10 examples](../examples/ch10_kv/)
on CPU with fixed arrays. The PDF export includes the code and its recorded
outputs. The written solutions also cover all four optional exercises.
