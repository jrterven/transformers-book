# The Evolution of the Transformer

**Juan Terven and Diana Córdova**

Companion repository for *The Evolution of the Transformer*, a historical and practical guide to Transformer architectures and their applications.

The book starts with the problems that led to attention and builds the original Transformer step by step. From there, it follows the development of language model families, modern Transformer blocks, efficient attention, state space models, and hybrid architectures. Each change is explained through the problem it addresses, a small worked example, and the tradeoffs it introduces.

Applications are a central part of the book. The discussion follows how Transformers have been adapted to language, images, speech, video, multimodal learning, time series, scientific data, and robotics. The emphasis is on understanding how a task is represented, what the model learns, and how its results should be evaluated.

## Topics

- Tokens, embeddings, attention, and the original encoder-decoder architecture.
- GPT, BERT, T5, and the development of pretrained language models.
- Normalization, gated feed-forward networks, positional representations, and modern decoder blocks.
- Training, inference, KV caches, sparse attention, and compressed representations.
- State space models, Mamba, and architectures that combine attention with recurrent state.
- Applications, Python implementations, and reproducible experiments.

## Python examples

The examples make the book's calculations visible through small implementations, explicit tensor shapes, worked numerical results, and reproducible checks.

Start with [Chapter 4: Self-Attention](examples/ch04_attention/). Its NumPy implementation covers query, key, and value projections; stable softmax; scaled dot-product attention; causal masks; independent batch and head axes; and a manual backward pass. The [chapter guide](examples/ch04_attention/README.md) explains the API, shapes, mask convention, and numerical example.

Run the following commands from this repository's root directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m examples.ch04_attention.run_examples
python -m unittest discover -s tests -v
```

The demo uses NumPy and runs on a CPU. It prints the three-token example, its causal version, tensor shapes, and a finite-difference check of the backward pass. The tests also check numerical references, masking, permutation equivariance, and isolation between examples and heads. See [local validation results](VALIDATION.md) for the versions and checks used.

The book's code listings are generated from the runnable source. After editing a marked section, regenerate the snippets and check that they are current:

```sh
python scripts/export_snippets.py
python scripts/export_snippets.py --check
```

## Background

The book assumes basic knowledge of Python, linear algebra, and neural networks. Attention and Transformer architectures are developed from the beginning.
