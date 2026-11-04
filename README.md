# The Evolution of the Transformer

**Juan Terven**

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

This repository accompanies the book's Python examples and experiments. Small implementations make the underlying calculations visible; application examples connect those calculations to complete tasks. Tensor shapes, numerical checks, evaluation, and computational requirements are part of the explanation.

## Background

The book assumes basic knowledge of Python, linear algebra, and neural networks. Attention and Transformer architectures are developed from the beginning.
