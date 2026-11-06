# Transformers

**Juan Terven and Diana Córdova**

<p align="center">
  <img src="assets/transformers-cover.jpg" alt="Transformers book cover by Juan Terven and Diana Córdova" width="360">
</p>

Companion repository for *Transformers*, a historical and practical guide to Transformer architectures and their applications.

The book starts with the problems that led to attention and builds the original Transformer step by step. From there, it follows the development of language model families, modern Transformer blocks, efficient attention, state space models, and hybrid architectures. Each change is explained through the problem it addresses, a small worked example, and the tradeoffs it introduces.

Applications are a central part of the book. The discussion follows how Transformers have been adapted to language, images, speech, video, multimodal learning, time series, scientific data, and robotics. The emphasis is on understanding how a task is represented, what the model learns, and how its results should be evaluated.

## Topics

- Tokens, embeddings, attention, and the original encoder-decoder architecture.
- GPT, BERT, T5, and the development of pretrained language models.
- Normalization, gated feed-forward networks, positional representations, and modern decoder blocks.
- Training compute, data allocation, and sparse mixtures of experts.
- Training, inference, KV caches, sparse attention, and compressed representations.
- State space models, Mamba, and architectures that combine attention with recurrent state.
- Applications, Python implementations, and reproducible experiments.

## Exercise solutions

Work through the [exercise solutions](solutions/) by chapter. The written
PDFs show the calculations step by step. The Python exercises are available
as executed Jupyter notebooks and as PDFs with the code and output.

## Python examples

The examples make the book's calculations visible through small implementations, explicit tensor shapes, worked numerical results, and reproducible checks.

Start with [Chapter 1: Tokens and Language Models](examples/ch01_language/). Its NumPy examples turn token IDs into embedding rows, combine an ordered two-token context, and turn vocabulary scores into probabilities. The matrices are chosen by hand so the forward pass can be checked with a calculator; this example does not train a language model.

```sh
python -m pip install -r requirements.txt
python -m examples.ch01_language.run_examples
```

[Chapter 2: Recurrence and Encoder-Decoder Models](examples/ch02_recurrence/) follows a scalar recurrent update and an LSTM memory cell, then implements an LSTM encoder-decoder without attention. Its optional training experiment reverses short color sequences and compares teacher-forced token predictions with complete greedy generations. The [chapter guide](examples/ch02_recurrence/README.md) explains state transfer, source lengths, BOS/EOS alignment, padding, and the local reference results.

```sh
python -m pip install -r requirements-ch02.txt
python -m examples.ch02_recurrence.run_examples
python -m examples.ch02_recurrence.run_examples --train --steps 750 --output output/ch02
```

[Chapter 3: Learning to Align](examples/ch03_alignment/) adds learned additive
attention to a recurrent encoder-decoder. It works through scores, source-axis
normalization, masks, weighted contexts, and generated-token alignment maps.
The experiment reuses Chapter 2's color-reversal split and records attention
for every generated token, including EOS.

```sh
python -m pip install -r requirements-ch03.txt
python -m examples.ch03_alignment.run_examples
python -m examples.ch03_alignment.run_examples --train --steps 750 --output output/ch03
```

[Chapter 4: Self-Attention](examples/ch04_attention/) then builds self-attention. Its NumPy implementation covers query, key, and value projections; stable softmax; scaled dot-product attention; causal masks; independent batch and head axes; and a manual backward pass. The [chapter guide](examples/ch04_attention/README.md) explains the API, shapes, mask convention, and numerical example.

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

The small model runs on a CPU and does not download pretrained weights or a dataset. Its forward and backward passes expose the architecture and check causal information flow; the randomly initialized outputs do not measure translation quality. The tests for Chapters 5–8 require PyTorch. With only the NumPy requirements installed, they are explicitly reported as skipped.

[Chapter 6: Training and Generation](examples/ch06_training/) trains that architecture on a reproducible sequence-reversal task. It covers shifted targets, padding-aware token loss, validation-selected checkpoints, greedy generation, and incremental decoding with self-attention and cross-attention KV caches. The [chapter guide](examples/ch06_training/README.md) explains the data split, training recipe, metrics, cache conventions, and timing boundaries.

```sh
python -m pip install -r requirements-ch06.txt
python -m examples.ch06_training.run_experiment --output output/ch06
python -m unittest discover -s tests -v
```

This example trains from scratch on one CPU thread. It writes a model checkpoint and an inspectable metrics record under `output/ch06/`. The generated results remain local; the source, tests, and exact commands are included here. The synthetic task makes training behavior visible and does not measure translation quality or general language ability.

[Chapter 7: GPT, BERT, and T5](examples/ch07_pretraining/) builds three prediction
tasks from short sentences. The examples make next-token shifts, selected masked
labels, missing spans, and encoder/decoder information paths visible. Two small
stacks with identical weights show what changes when a later input is hidden
by a causal mask or made available through bidirectional attention. The
[chapter guide](examples/ch07_pretraining/README.md) explains the arrays, code,
teaching conventions, and optional fixed-example training check.

```sh
python -m pip install -r requirements-ch05.txt
python -m examples.ch07_pretraining.run_examples
python -m examples.ch07_pretraining.run_examples --train --steps 200 --output output/ch07.json
```

These models reuse Chapter 5's small blocks, start with random weights, and run
without downloads. They expose architectural ideas and objectives; they are
not replicas of the pretrained GPT, BERT, or T5 releases. The optional run
memorizes one example per objective to check the wiring, and does not compare
the families' general language abilities.

[Chapter 8: From the Original Block to Modern Decoders](examples/ch08_modern/)
builds a small causal decoder from RMSNorm, SwiGLU, and rotary query and key
positions. The [chapter guide](examples/ch08_modern/README.md) follows the
hand calculations, then checks parameter counts, causal and padding behavior,
position shifts, and a next-token update on a CPU.

```sh
python -m pip install -r requirements-ch05.txt
python -m examples.ch08_modern.run_examples --output output/ch08.json
```

[Chapter 9: Scaling Data, Compute, and Expert Capacity](examples/ch09_scaling/)
compares dense training allocations under the same estimated compute budget.
It then follows a token through a router and shows how a batch can overflow
one expert while others still have room. The [chapter guide](examples/ch09_scaling/README.md)
explains the units, array shapes, and expected results.

```sh
python -m pip install -r requirements.txt
python -m examples.ch09_scaling.run_examples --output output/ch09.json
```

[Chapter 10: Incremental Decoding and KV Representations](examples/ch10_kv/)
compares a full causal pass with prefill and cached decoding. The examples
count the stored keys and values for MHA, GQA, and MQA, then compare two
equivalent ways to read a latent cache. See the [chapter guide](examples/ch10_kv/README.md)
for the array shapes and expected results.

```sh
python -m pip install -r requirements.txt
python -m examples.ch10_kv.run_examples --output output/ch10.json
```

[Chapter 11: Long Context, Sparse Access, and Sequence Compression](examples/ch11_long_context/)
builds local and global masks, ranks and gathers entries before main
attention, and forms summaries over completed blocks. The [chapter guide](examples/ch11_long_context/README.md)
explains the causal checks and the example that reads a summary beside recent raw entries.

```sh
python -m pip install -r requirements.txt
python -m examples.ch11_long_context.run_examples --output output/ch11.json
```

[Chapter 12: Efficient Execution](examples/ch12_efficient/) follows an online
softmax update and compares tiled attention with a full-matrix reference. The
[chapter guide](examples/ch12_efficient/README.md) explains the running state,
causal positions, and temporary score-array sizes.

```sh
python -m pip install -r requirements.txt
python -m examples.ch12_efficient.run_examples --output output/ch12.json
```

[Chapter 13: Structured State Spaces and Long Convolutions](examples/ch13_state_spaces/)
runs the same linear state space model as a recurrence and as a convolution.
The [chapter guide](examples/ch13_state_spaces/README.md) follows the filter
coefficients, FFT padding, and a gated convolution inspired by Hyena.

```sh
python -m pip install -r requirements.txt
python -m examples.ch13_state_spaces.run_examples --output output/ch13.json
```

[Chapter 14: Selective State Spaces](examples/ch14_selective_ssm/) follows
input-dependent state updates, then evaluates one SSD head as a recurrence,
as a dense matrix, and in chunks. The
[chapter guide](examples/ch14_selective_ssm/README.md) explains the retain
and write coefficients and the state carried between chunks.

```sh
python -m pip install -r requirements.txt
python -m examples.ch14_selective_ssm.run_examples --output output/ch14.json
```

[Chapter 15: Delta Rules and Hybrid Stacks](examples/ch15_hybrids/) follows
a delta-rule correction and a token moving through a recurrent–attention–recurrent
stack. The [chapter guide](examples/ch15_hybrids/README.md) shows the layer
traces and how the state and KV cache sizes change with the mixer schedule.

```sh
python -m pip install -r requirements.txt
python -m examples.ch15_hybrids.run_examples --output output/ch15.json
```

[Chapter 16: From Pretrained Weights to Task Behavior](examples/ch16_adaptation/)
compares a frozen feature extractor with full fine-tuning, aligns response
targets, and checks low-rank adaptation and the DPO objective. The
[chapter guide](examples/ch16_adaptation/README.md) explains the trainable
parameters, the response mask, and the adapter checkpoint roundtrip.

```sh
python -m pip install -r requirements-ch05.txt
python -m examples.ch16_adaptation.run_examples --output output/ch16.json
```

[Chapter 17: Designing Reproducible Evaluations](examples/ch17_evaluation/)
splits related examples by group, computes binary metrics and reliability
bins, and compares paired predictions. The
[chapter guide](examples/ch17_evaluation/README.md) explains the resampling
unit and the scope of the resulting interval.

```sh
python -m pip install -r requirements.txt
python -m examples.ch17_evaluation.run_examples --output output/ch17.json
```

[Chapter 18: Language Understanding](examples/ch18_understanding/) pools
token features for a sentence classifier, aligns labels to first subwords,
and checks exact entity spans. The
[chapter guide](examples/ch18_understanding/README.md) explains the masks,
word indices, and BIO decoding policy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch18_understanding.run_examples --output output/ch18.json
```

Tests cover numerical references, masking, permutation behavior, independent heads and layers, normalization, gradients, and complete-model prefix and padding checks. See [local validation results](VALIDATION.md) for the versions and checks used.

The book's code listings are generated from the runnable source. After editing a marked section, regenerate the snippets and check that they are current:

```sh
python scripts/export_snippets.py
python scripts/export_snippets.py --check
```

## Background

The book assumes basic knowledge of Python, linear algebra, and neural networks. Attention and Transformer architectures are developed from the beginning.
