# Transformers

**Juan Terven and Diana Córdova**

Companion repository for *Transformers*, a historical and practical guide to Transformer architectures and their applications.

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

The Chapter 4 demo uses NumPy and runs on a CPU. It prints the three-token example, its causal version, tensor shapes, and a finite-difference check of the backward pass.

[Chapter 5: The Complete Transformer](examples/ch05_transformer/) assembles multi-head attention, sinusoidal positions, positionwise feed-forward networks, residual connections, and post-norm encoder and decoder layers in PyTorch. The [chapter guide](examples/ch05_transformer/README.md) describes the model, masks, shapes, and deterministic numerical examples. Run it in the same environment after installing the additional requirements:

```sh
python -m pip install -r requirements-ch05.txt
python -m examples.ch05_transformer.run_examples
python -m unittest discover -s tests -v
```

The small model runs on a CPU and does not download pretrained weights or a dataset. Its forward and backward passes expose the architecture and check causal information flow; the randomly initialized outputs do not measure translation quality. Chapter 5 tests require PyTorch. With only the Chapter 4 requirements installed, they are explicitly reported as skipped.

Tests cover numerical references, masking, permutation behavior, independent heads and layers, normalization, gradients, and complete-model prefix and padding checks. See [local validation results](VALIDATION.md) for the versions and checks used.

The book's code listings are generated from the runnable source. After editing a marked section, regenerate the snippets and check that they are current:

```sh
python scripts/export_snippets.py
python scripts/export_snippets.py --check
```

## Background

The book assumes basic knowledge of Python, linear algebra, and neural networks. Attention and Transformer architectures are developed from the beginning.
