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

Start with the [12 Chapter 1 exercises](examples/ch01_language/EXERCISES.md),
ordered from foundational questions to two challenges. The written solutions,
executed notebook, and notebook PDF all use the same numbering.

## Python examples

The examples make the book's calculations visible through small implementations, explicit tensor shapes, worked numerical results, and reproducible checks.

Start with [Chapter 1: Tokens and Language Models](examples/ch01_language/). Its NumPy examples turn token IDs into embedding rows, combine an ordered two-token context, and turn vocabulary scores into probabilities. The matrices are chosen by hand so the forward pass can be checked with a calculator; this example does not train a language model.

```sh
python -m pip install -r requirements.txt
python -m examples.ch01_language.run_examples
```

[Chapter 2: Recurrence and Encoder-Decoder Models](examples/ch02_recurrence/) follows a scalar recurrent update and an LSTM memory cell, then implements an LSTM encoder-decoder without attention. Its optional training experiment reverses short color sequences and compares teacher-forced token predictions with complete greedy generations. The [chapter guide](examples/ch02_recurrence/README.md) explains state transfer, source lengths, BOS/EOS alignment, padding, and the local reference results. The [14 progressive exercises](examples/ch02_recurrence/EXERCISES.md) have matching [worked solutions and executed notebooks](solutions/ch02_recurrence/README.md).

```sh
python -m pip install -r requirements-ch02.txt
python -m examples.ch02_recurrence.run_examples
python -m examples.ch02_recurrence.run_examples --train --steps 750 --output output/ch02
```

[Chapter 3: Learning to Align](examples/ch03_alignment/) adds learned additive
attention to a recurrent encoder-decoder. It works through scores, source-axis
normalization, masks, weighted contexts, and generated-token alignment maps.
The experiment reuses Chapter 2's color-reversal split and records attention
for every generated token, including EOS. The [14 exercises](examples/ch03_alignment/EXERCISES.md)
progress from worked reads to implementation checks and an optional gradient
challenge. [Matching solutions](solutions/ch03_alignment/) include an executed
notebook and two PDFs generated from it.

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
python -m unittest discover -s tests -p test_attention.py -v
```

The Chapter 4 demo uses NumPy and runs on a CPU. It prints the three-token example, its causal version, tensor shapes, value and future-input interventions, and a finite-difference check of the backward pass. Its [15 exercises](examples/ch04_attention/EXERCISES.md) progress from hand calculations to information-flow checks and optional gradients. [Complete solutions](solutions/ch04_attention/README.md) include an executed notebook and two matching PDFs.

[Chapter 5: The Complete Transformer](examples/ch05_transformer/) assembles multi-head attention, sinusoidal positions, positionwise feed-forward networks, residual connections, and post-norm encoder and decoder layers in PyTorch. The [chapter guide](examples/ch05_transformer/README.md) describes the model, masks, shapes, and deterministic numerical examples. Run it in the same environment after installing the additional requirements:

```sh
python -m pip install -r requirements-ch05.txt
python -m examples.ch05_transformer.run_examples
python -m unittest discover -s tests -v
```

All [15 Chapter 5 exercises](examples/ch05_transformer/EXERCISES.md) have matching [written and executable solutions](solutions/ch05_transformer/README.md). The small model runs on a CPU and does not download pretrained weights or a dataset. Its forward and backward passes expose the architecture and check causal information flow; the randomly initialized outputs do not measure translation quality. The tests for Chapters 5–8 require PyTorch. With only the NumPy requirements installed, they are explicitly reported as skipped.

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

[Chapter 19: Translation and Summarization](examples/ch19_generation/)
builds shifted text pairs, compares greedy and beam decoding, and computes
clipped precision and subsequence overlap. The
[chapter guide](examples/ch19_generation/README.md) explains the completion
flag and the optional encoder-decoder training step.

```sh
python -m pip install -r requirements.txt
python -m examples.ch19_generation.run_examples --output output/ch19.json
python -m pip install -r requirements-ch06.txt
python -m examples.ch19_generation.run_examples --train-step
```

[Chapter 20: Search, Retrieval, and Grounded Assistants](examples/ch20_retrieval/)
ranks a local passage collection with BM25 and dense scores, reranks a
candidate set, and checks what survives the context budget. The
[chapter guide](examples/ch20_retrieval/README.md) explains the distinction
between literal evidence, annotated recall, and answer support.

```sh
python -m pip install -r requirements.txt
python -m examples.ch20_retrieval.run_examples --output output/ch20.json
```

[Chapter 21: Image and Video Understanding](examples/ch21_vision/) extracts
image patches, fits linear probes on frozen features, and compares joint
and divided video attention counts. The
[chapter guide](examples/ch21_vision/README.md) follows the patch order
and the held-out synthetic images.

```sh
python -m pip install -r requirements.txt
python -m examples.ch21_vision.run_examples --output output/ch21.json
```

[Chapter 22: Object Detection and Segmentation](examples/ch22_detection/)
computes box and mask overlap, matches object slots to annotations, and
scores ranked detections. The [chapter guide](examples/ch22_detection/README.md)
explains the coordinate convention and the single-threshold AP example.

```sh
python -m pip install -r requirements.txt
python -m examples.ch22_detection.run_examples --output output/ch22.json
```

[Chapter 23: Vision–Language and Multimodal Models](examples/ch23_multimodal/)
computes a contrastive similarity matrix, scores retrieval with multiple
relevant items, and projects cross-attention outputs to a language width.
The [chapter guide](examples/ch23_multimodal/README.md) explains the two
retrieval directions and the visual bridge.

```sh
python -m pip install -r requirements.txt
python -m examples.ch23_multimodal.run_examples --output output/ch23.json
```

[Chapter 24: Speech and Audio](examples/ch24_speech_audio/) builds an
overlapping-frame spectrogram, labels clean and noisy synthetic tones,
and scores word edits. The [chapter guide](examples/ch24_speech_audio/README.md)
explains the frame axes and the convention for transcripts without reference words.

```sh
python -m pip install -r requirements.txt
python -m examples.ch24_speech_audio.run_examples --output output/ch24.json
```

[Chapter 25: Image and Video Generation](examples/ch25_visual_generation/)
mixes a clean grid with noise, follows a deterministic diffusion sampler,
and compares conditional and unconditional predictions. The
[chapter guide](examples/ch25_visual_generation/README.md) explains the
four-value fixture and the two sampler outputs.

```sh
python -m pip install -r requirements.txt
python -m examples.ch25_visual_generation.run_examples --output output/ch25.json
```

[Chapter 26: Models for Code](examples/ch26_code_models/) formats an
infilling prompt and runs checked-in candidate functions against smoke and
boundary tests. The [chapter guide](examples/ch26_code_models/README.md)
explains the case records and the per-task pass@k calculation.

```sh
python -m examples.ch26_code_models.run_examples --output output/ch26.json
```

[Chapter 27: Time-Series Forecasting](examples/ch27_time_series/) builds
windows in time order, compares seasonal forecasts with and without
fitted drift, and measures interval coverage by horizon. The
[chapter guide](examples/ch27_time_series/README.md) explains which
observations the forecaster can use at each origin.

```sh
python -m pip install -r requirements.txt
python -m examples.ch27_time_series.run_examples --output output/ch27.json
```

[Chapter 28: Learning from Tables](examples/ch28_tabular/) fits an
encoding on training rows, weights labeled context rows, and checks
binary probability predictions. The [chapter guide](examples/ch28_tabular/README.md)
explains the missing and unknown indicators and the Brier score.

```sh
python -m pip install -r requirements.txt
python -m examples.ch28_tabular.run_examples --output output/ch28.json
```

[Chapter 29: Scientific Sequences and Biological Structure](examples/ch29_protein/)
counts residues and ordered pairs, pools supplied residue vectors, and
builds a contact map from four coordinates. The
[chapter guide](examples/ch29_protein/README.md) explains the alphabet,
mask, and distance convention.

```sh
python -m pip install -r requirements.txt
python -m examples.ch29_protein.run_examples --output output/ch29.json
```

[Chapter 30: Decisions and Embodied Action](examples/ch30_decisions/)
builds return-to-go targets, fits two count policies, and runs them in a
small grid. The [chapter guide](examples/ch30_decisions/README.md)
explains the trajectory format and how the shifted trap affects a rollout.

```sh
python -m examples.ch30_decisions.run_examples --output output/ch30.json
```

[Chapter 31: What Changed, and What Remains?](examples/ch31_perspective/)
counts causal attention pairs and compares four ways to retain a
stream of delivery events. The [chapter guide](examples/ch31_perspective/README.md)
explains the two questions and the units used to record stored information.

```sh
python -m examples.ch31_perspective.run_examples --output output/ch31.json
```

## Examples for the appendices

The reference appendices use calculations from the chapter examples.

| Appendix | Companion examples |
| --- | --- |
| A. Notation and tensor shapes | [Self-attention and masks](examples/ch04_attention/) |
| B. Calculus and optimization | [Training the small Transformer](examples/ch06_training/) |
| C. Stable numerics | [Stable softmax](examples/ch04_attention/) and [normalization](examples/ch08_modern/) |
| D. Complexity and memory | [KV accounting](examples/ch10_kv/) and [causal pair counts](examples/ch31_perspective/) |
| E. Convolution and scans | [Structured state spaces](examples/ch13_state_spaces/) and [selective updates](examples/ch14_selective_ssm/) |
| F. Diffusion foundations | [Noise, guidance, and DDIM](examples/ch25_visual_generation/) |

Tests cover numerical references, masking, permutation behavior, independent heads and layers, normalization, gradients, and complete-model prefix and padding checks. See [local validation results](VALIDATION.md) for the versions and checks used.

The book's code listings are generated from the runnable source. After editing a marked section, regenerate the snippets and check that they are current:

```sh
python scripts/export_snippets.py
python scripts/export_snippets.py --check
```

## Background

The book assumes basic knowledge of Python, linear algebra, and neural networks. Attention and Transformer architectures are developed from the beginning.
