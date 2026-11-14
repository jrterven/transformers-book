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

Chapter 1 also includes [reproducible observations of a pretrained language model](examples/ch01_language/PRETRAINED_FIGURES.md): actual token pieces and IDs, fixed lookup rows versus context-dependent hidden vectors, and full-vocabulary next-token probabilities. These optional figures use a pinned DistilGPT2 checkpoint.

[Chapter 2: Recurrence and Encoder-Decoder Models](examples/ch02_recurrence/) follows a scalar recurrent update and an LSTM memory cell, then implements an LSTM encoder-decoder without attention. Its optional training experiment reverses short color sequences and compares teacher-forced token predictions with complete greedy generations. The [chapter guide](examples/ch02_recurrence/README.md) explains state transfer, source lengths, BOS/EOS alignment, padding, and the local reference results. [Measured model figures](examples/ch02_recurrence/TRAINED_FIGURES.md) expose its actual states, learned gates, and reference-conditioned versus freely generated predictions. The [14 progressive exercises](examples/ch02_recurrence/EXERCISES.md) have matching [worked solutions and executed notebooks](solutions/ch02_recurrence/README.md).

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

The [measured attention figures](examples/ch03_alignment/TRAINED_FIGURES.md)
expose the trained model's raw scores, source weights, and vector contributions,
then isolate the effects of source padding and decoder input timing with
controlled interventions.

```sh
python -m pip install -r requirements-ch03.txt
python -m examples.ch03_alignment.run_examples
python -m examples.ch03_alignment.run_examples --train --steps 750 --output output/ch03
```

[Chapter 4: Self-Attention](examples/ch04_attention/) then builds self-attention. Its NumPy implementation covers query, key, and value projections; stable softmax; scaled dot-product attention; causal masks; independent batch and head axes; and a manual backward pass. The [chapter guide](examples/ch04_attention/README.md) explains the API, shapes, mask convention, and numerical example.

The [pretrained head figures](examples/ch04_attention/PRETRAINED_FIGURES.md)
open a real DistilGPT2 head into its projected features, scores, weights, and
value contributions, then test the causal information rule by changing one
future input token. Full observations and numerical reconstruction checks
connect these images to the NumPy lesson.

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

The optional [Chapter 5 trained-model inspection](examples/ch05_transformer/TRAINED_FIGURES.md) regenerates three measured figures from Chapter 6's color-reversal checkpoint. It reconstructs learned head contributions, residual addition and LayerNorm, and a ReLU FFN, checking each against the native module output. Its plotting dependencies are in `requirements-ch05-figures.txt`.

[Chapter 6: Training and Generation](examples/ch06_training/) trains that architecture on a reproducible sequence-reversal task. It covers shifted targets, padding-aware token loss, validation-selected checkpoints, greedy generation, and incremental decoding with self-attention and cross-attention KV caches. The [chapter guide](examples/ch06_training/README.md) explains the data split, training recipe, metrics, cache conventions, and timing boundaries.

Its [measured training and generation figures](examples/ch06_training/TRAINED_FIGURES.md) follow a real minibatch gradient into Adam's first update, compare a fixed validation example at three training stages, and count actual KV tensor storage during generation. The replay is checked against the saved checkpoint, and cached logits are compared with full-prefix logits at every step.

```sh
python -m pip install -r requirements-ch06.txt
python -m examples.ch06_training.run_experiment --output output/ch06
python -m unittest discover -s tests -p test_training.py -v
```

All [15 exercises](examples/ch06_training/EXERCISES.md) have matching [written and executable solutions](solutions/ch06_training/README.md).

This example trains from scratch on one CPU thread. It writes a model checkpoint and an inspectable metrics record under `output/ch06/`. New run outputs remain local. A [reference metrics record](examples/ch06_training/reference_metrics.json), source, tests, and exact commands are included here. The synthetic task makes training behavior visible and does not measure translation quality or general language ability.

[Chapter 7: GPT, BERT, and T5](examples/ch07_pretraining/) builds three prediction
tasks from short sentences. The examples make next-token shifts, selected masked
labels, missing spans, and encoder/decoder information paths visible. Two small
stacks with identical weights show what changes when a later input is hidden
by a causal mask or made available through bidirectional attention. The
[chapter guide](examples/ch07_pretraining/README.md) explains the arrays, code,
teaching conventions, and optional fixed-example training check. All
[14 exercises](examples/ch07_pretraining/EXERCISES.md) have
[complete written and executable solutions](solutions/ch07_pretraining/README.md),
including the distinction between input replacement, attention permissions,
and loss selection.

The optional [pretrained GPT-2 and BERT inspection](examples/ch07_pretraining/PRETRAINED_FIGURES.md)
adds six reproducible figures from pinned public checkpoints. It follows token
and segment inputs, vocabulary probabilities, greedy feedback, and controlled
context changes through all hidden stages. Independent calculations check the
native output heads and BERT input embeddings. These observations use learned
models alongside the small objective examples; they are not a benchmark of
the two families.

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

The [14 progressive exercises](examples/ch08_modern/EXERCISES.md) and
[complete solutions](solutions/ch08_modern/README.md) connect hand calculations
to executable checks, including multiple RoPE frequencies, padding masks,
position offsets, and a finite-difference gradient comparison.

The optional [Chapter 8 pretrained inspection](examples/ch08_modern/PRETRAINED_FIGURES.md)
adds four measured SmolLM2 figures. It reconstructs RMSNorm and SwiGLU,
follows native rotary pairs through masked softmax, and measures residual
updates through all 30 blocks. The guide distinguishes the checkpoint's
split-half rotary layout and shared key--value heads from the teaching model.


[Chapter 9: Scaling Data, Compute, and Expert Capacity](examples/ch09_scaling/)
compares dense training allocations under the same estimated compute budget.
It then follows a token through a router and shows how a batch can overflow
one expert while others still have room. The [chapter guide](examples/ch09_scaling/README.md)
explains the units, array shapes, and expected results.

```sh
python -m pip install -r requirements.txt
python -m examples.ch09_scaling.run_examples --output output/ch09.json
```

The [14 progressive exercises](examples/ch09_scaling/EXERCISES.md) and
[complete solutions](solutions/ch09_scaling/README.md) connect budget arithmetic
to expert routing, including the difference between a balance statistic and
actual load, and between Switch gates and selected-logit normalization.

The optional [Chapter 9 pretrained inspection](examples/ch09_scaling/PRETRAINED_FIGURES.md)
follows a fixed token through a real Switch-base-8 router, ReLU expert, and
residual addition. Four measured figures also reveal overflow under a local
capacity change and distinguish stored weights from selected matrix entries.

[Chapter 10: Incremental Decoding and KV Representations](examples/ch10_kv/)
compares a full causal pass with prefill and cached decoding. The examples
count the stored keys and values for MHA, GQA, and MQA, then compare two
equivalent ways to read a latent cache. See the [chapter guide](examples/ch10_kv/README.md)
for the array shapes and expected results.

The optional [Chapter 10 pretrained inspection](examples/ch10_kv/PRETRAINED_FIGURES.md)
measures actual SmolLM2 and GPT-2 caches. Four figures follow saved vectors,
shared-history reads, equivalent replay, and exact payload growth, including
a controlled demonstration of a RoPE-position reset.

```sh
python -m pip install -r requirements.txt
python -m examples.ch10_kv.run_examples --output output/ch10.json
```

The [12 progressive exercises](examples/ch10_kv/EXERCISES.md) and
[complete solutions](solutions/ch10_kv/README.md) connect cache mechanics to
causality, actual storage, and latent attention. Every exercise is included
in the notebook and both PDF editions.

[Chapter 11: Long Context, Sparse Access, and Sequence Compression](examples/ch11_long_context/)
builds local and global masks, ranks and gathers entries before main
attention, and forms summaries over completed blocks. The [chapter guide](examples/ch11_long_context/README.md)
explains the causal checks and the example that reads a summary beside recent raw entries. The [twelve exercises](examples/ch11_long_context/EXERCISES.md) have [complete solutions](solutions/ch11_long_context/README.md) in an executed notebook and two PDF editions.

The optional [Chapter 11 pretrained inspection](examples/ch11_long_context/PRETRAINED_FIGURES.md)
observes real Longformer windows and global relays. Four figures compare
native probabilities, trace a single-token change across two blocks, and
measure the effect of retaining and renormalizing fewer attention entries.

```sh
python -m pip install -r requirements.txt
python -m examples.ch11_long_context.run_examples --output output/ch11.json
```

[Chapter 12: Efficient Execution](examples/ch12_efficient/) follows an online
softmax state across score tiles, checks exact attention within floating-point
tolerance, and distinguishes score workspace from memory traffic and runtime.

The optional [Chapter 12 pretrained inspection](examples/ch12_efficient/PRETRAINED_FIGURES.md)
uses real SmolLM2 vectors to trace online softmax, then compares native eager
attention with CPU SDPA. Four figures separate numerical agreement, actual
operator allocations, and measured first-attention-module runtime.

The [eleven exercises](examples/ch12_efficient/EXERCISES.md) have
[complete solutions in every edition](solutions/ch12_efficient/README.md),
with 16 executed code cells and two inspectable diagrams.

```sh
python -m examples.ch12_efficient.run_examples --output output/ch12.json
```

[Chapter 13: Structured State Spaces and Long Convolutions](examples/ch13_state_spaces/)
compares recurrent, direct-convolution, and FFT evaluations of the same linear
system. It traces saved history, circular wraparound, and gates placed before
or after a filter. The expanded chapter also derives discretization and
explains HiPPO, structured S4 dynamics, and implicit Hyena filters. Its optional
[published-model inspection](examples/ch13_state_spaces/PRETRAINED_FIGURES.md)
produces six measured figures from a trained SaShiMi S4 layer and tiny
HyenaDNA, with independent numerical reconstruction and a causal perturbation.
The [eleven exercises](examples/ch13_state_spaces/EXERCISES.md)
have [complete solutions in every edition](solutions/ch13_state_spaces/README.md),
with 17 executed code cells and two diagnostic plots.

```sh
python -m examples.ch13_state_spaces.run_examples --output output/ch13.json
```

[Chapter 14: Selective State Spaces](examples/ch14_selective_ssm/) follows
input-dependent retain, write, and read operations, derives a learnable gate
and ordered scans, then compares recurrent, dense, and chunked SSD with both
scalar inputs and matrix-valued head states. The optional
[published-model inspection](examples/ch14_selective_ssm/PRETRAINED_FIGURES.md)
adds six measured figures from Mamba and Mamba-2, with native control arrays,
the authors' SSD reference, cache payloads, and a causal intervention.
The [ten exercises](examples/ch14_selective_ssm/EXERCISES.md)
have [complete solutions in every edition](solutions/ch14_selective_ssm/README.md),
including 21 executed code cells, two plots, and a controlled memory-task design.

```sh
python -m examples.ch14_selective_ssm.run_examples --output output/ch14.json
```

[Chapter 15: Delta Rules and Hybrid Stacks](examples/ch15_hybrids/) traces
prediction-error writes, interfering key directions, triangular chunk systems,
and separate per-layer states and caches. Its optional
[published-model inspection](examples/ch15_hybrids/PRETRAINED_FIGURES.md)
reproduces six measured figures from the Qwen3.5-0.8B-Base language pathway.
Its [ten exercises](examples/ch15_hybrids/EXERCISES.md) have
[complete solutions in all editions](solutions/ch15_hybrids/README.md), with
21 executed notebook cells and two plots.

```sh
python -m examples.ch15_hybrids.run_examples --output output/ch15.json
```

[Chapter 16: From Pretrained Weights to Task Behavior](examples/ch16_adaptation/)
compares a frozen feature extractor with full fine-tuning, aligns response
targets, and checks low-rank adaptation and the DPO objective. The
[chapter guide](examples/ch16_adaptation/README.md) explains the trainable
parameters, the response mask, and the adapter checkpoint roundtrip.
The optional [pretrained-model experiment](examples/ch16_adaptation/PRETRAINED_FIGURES.md)
actually trains a frozen-feature head, SFT/LoRA adapters, and DPO adapters on
fictional requests, producing six measured figures and reload/merge checks.
The [twelve exercises](examples/ch16_adaptation/EXERCISES.md) have complete
[worked solutions](solutions/ch16_adaptation/README.md) in an executed
notebook and both PDF editions, including batch averaging, frozen gradient
paths and checkpoint probes.

```sh
python -m pip install -r requirements-ch05.txt
python -m examples.ch16_adaptation.run_examples --output output/ch16.json
```

[Chapter 17: Designing Reproducible Evaluations](examples/ch17_evaluation/)
splits related examples by group, computes binary metrics and reliability
bins, and compares paired predictions. The chapter's
[measured SmolLM2 evaluations](examples/ch17_evaluation/PRETRAINED_FIGURES.md)
use the Chapter 16 outputs and 200 new prompts to show how an answer rule,
the choice of resampling unit, and the location of evidence affect the result.
The [twelve exercises](examples/ch17_evaluation/EXERCISES.md) have complete
[worked solutions](solutions/ch17_evaluation/README.md) in an executed
notebook and both PDF editions, including error-cost decisions and explicit
long-context placement controls.

```sh
python -m pip install -r requirements.txt
python -m examples.ch17_evaluation.run_examples --output output/ch17.json
```

[Chapter 18: Language Understanding](examples/ch18_understanding/) pools
token features for a sentence classifier, aligns labels to first subwords,
and checks exact entity spans. The
[measured examples](examples/ch18_understanding/PRETRAINED_FIGURES.md)
include trained probes on BERT, a released sentence-pair model, and a paired
casing test with DistilBERT for named entity recognition.
The [twelve exercises](examples/ch18_understanding/EXERCISES.md) have complete
[worked solutions](solutions/ch18_understanding/README.md) in an executed
notebook and both PDF editions, including micro aggregation and class-prevalence effects.

```sh
python -m pip install -r requirements.txt
python -m examples.ch18_understanding.run_examples --output output/ch18.json
```

[Chapter 19: Translation and Summarization](examples/ch19_generation/)
builds shifted text pairs, compares greedy and beam decoding, and computes
clipped precision and subsequence overlap. The chapter also follows an actual
OPUS-MT translation token by token and tests how BART summaries respond to
the available source text and output limits.
[Pretrained demonstrations and figures](examples/ch19_generation/PRETRAINED_FIGURES.md).

The
[chapter guide](examples/ch19_generation/README.md) explains the completion
flag and the optional encoder-decoder training step. The [12 exercises](examples/ch19_generation/EXERCISES.md)
and [complete worked solutions](solutions/ch19_generation/README.md) add an
LCS table, a corpus-aggregation comparison, and a source-support audit.

```sh
python -m pip install -r requirements.txt
python -m examples.ch19_generation.run_examples --output output/ch19.json
python -m pip install -r requirements-ch06.txt
python -m examples.ch19_generation.run_examples --train-step
```

[Chapter 20: Search, Retrieval, and Grounded Assistants](examples/ch20_retrieval/)
ranks a local passage collection with BM25 and dense scores, reranks a
candidate set, and checks what survives the context budget. A measured run
follows Qwen3 retrieval and reranking into Qwen3.5 answers and citations.
[Measured RAG demonstrations and figures](examples/ch20_retrieval/PRETRAINED_FIGURES.md).

The
[chapter guide](examples/ch20_retrieval/README.md) explains the distinction
between literal evidence, annotated recall, and answer support. The [12 exercises](examples/ch20_retrieval/EXERCISES.md)
and [complete solutions](solutions/ch20_retrieval/README.md) add worked
probability mixtures and visual checks of evidence lost during packing.

```sh
python -m pip install -r requirements.txt
python -m examples.ch20_retrieval.run_examples --output output/ch20.json
```

[Chapter 21: Image and Video Understanding](examples/ch21_vision/) extracts
image patches, fits linear probes on frozen features, and compares joint
and divided video attention counts. The
[measured visual demonstrations](examples/ch21_vision/PRETRAINED_FIGURES.md)
trace DINOv3 patch vectors and test V-JEPA 2 on reversed video clips. The
[chapter guide](examples/ch21_vision/README.md) follows the patch order
and the held-out synthetic images. The [12 exercises](examples/ch21_vision/EXERCISES.md)
and [complete solutions](solutions/ch21_vision/README.md) trace pixels into
tokens, explain position information, and design video evaluation protocols.

```sh
python -m pip install -r requirements.txt
python -m examples.ch21_vision.run_examples --output output/ch21.json
```

[Chapter 22: Object Detection and Segmentation](examples/ch22_detection/)
computes box and mask overlap, matches object slots to annotations, and
scores ranked detections. The [chapter guide](examples/ch22_detection/README.md)
explains the coordinate convention, the single-threshold AP example, and
the measured RF-DETR and SAM 2.1 runs on a COCO photograph. The
[11 exercises](examples/ch22_detection/EXERCISES.md) and
[complete solutions](solutions/ch22_detection/README.md) add mask diagrams,
precision envelopes and coordinate checks.

```sh
python -m pip install -r requirements.txt
python -m examples.ch22_detection.run_examples --output output/ch22.json
```

[Chapter 23: Vision–Language and Multimodal Models](examples/ch23_multimodal/)
computes a contrastive similarity matrix, scores retrieval with multiple
relevant items, and projects cross-attention outputs to a language width.
The [chapter guide](examples/ch23_multimodal/README.md) explains the two
retrieval directions and the visual bridge. It also reproduces the image-text
rankings and visual questions with SigLIP 2 and Qwen. The [11 exercises](examples/ch23_multimodal/EXERCISES.md)
and [complete solutions](solutions/ch23_multimodal/README.md) add probability
heatmaps, recall curves and a paired spatial test.

```sh
python -m pip install -r requirements.txt
python -m examples.ch23_multimodal.run_examples --output output/ch23.json
```

[Chapter 24: Speech and Audio](examples/ch24_speech_audio/) builds an
overlapping-frame spectrogram, labels clean and noisy synthetic tones,
and scores word edits. The [chapter guide](examples/ch24_speech_audio/README.md)
explains the frame axes and the convention for transcripts without reference words.
The [11 exercises](examples/ch24_speech_audio/EXERCISES.md) and
[complete solutions](solutions/ch24_speech_audio/README.md) add window,
spectrum and caption-timing plots with executed checks.

```sh
python -m pip install -r requirements.txt
python -m examples.ch24_speech_audio.run_examples --output output/ch24.json
```

[Chapter 25: Image and Video Generation](examples/ch25_visual_generation/)
mixes a clean grid with noise, follows a deterministic diffusion sampler,
and compares conditional and unconditional predictions. The
[chapter guide](examples/ch25_visual_generation/README.md) explains the
four-value fixture and the two sampler outputs.
The [11 exercises](examples/ch25_visual_generation/EXERCISES.md) and
[complete solutions](solutions/ch25_visual_generation/README.md) add
error propagation, guidance geometry and separate quality/diversity checks.

```sh
python -m pip install -r requirements.txt
python -m examples.ch25_visual_generation.run_examples --output output/ch25.json
```

[Chapter 26: Models for Code](examples/ch26_code_models/) formats an
infilling prompt and runs checked-in candidate functions against smoke and
boundary tests. The [chapter guide](examples/ch26_code_models/README.md)
also reproduces the measured infilling, code generation and repository repair
demos with local checkpoints and explains the per-task pass@k calculation.

The [11 exercises](examples/ch26_code_models/EXERCISES.md) and
[complete solutions](solutions/ch26_code_models/README.md) distinguish
per-case accuracy, complete-candidate success and task-level pass@k.

```sh
python -m examples.ch26_code_models.run_examples --output output/ch26.json
```

[Chapter 27: Time-Series Forecasting](examples/ch27_time_series/) builds
windows in time order, compares seasonal forecasts with and without
fitted drift, and measures interval coverage by horizon. The
[chapter guide](examples/ch27_time_series/README.md) also reproduces
TimesFM 3 and Timestral forecasts of hourly electricity demand in New York
and Texas, with aligned seasonal baselines and a local-calendar comparison.

The [11 exercises](examples/ch27_time_series/EXERCISES.md) and
[complete solutions](solutions/ch27_time_series/README.md) distinguish
information availability, seasonal drift, point error and interval calibration.

```sh
python -m pip install -r requirements.txt
python -m examples.ch27_time_series.run_examples --output output/ch27.json
```

[Chapter 28: Learning from Tables](examples/ch28_tabular/) fits an
encoding on training rows and works through binary probability scoring
and regression intervals. The [chapter guide](examples/ch28_tabular/README.md)
includes frozen TabICLv2 and Mitra-v2 demos on UCI Raisin classification
and recipe-disjoint concrete-strength regression. It includes the
exact splits, saved predictions, and commands to redraw the figures.

The [11 exercises](examples/ch28_tabular/EXERCISES.md) and
[complete solutions](solutions/ch28_tabular/README.md) explain missing and
unseen values, context weighting, calibration and deployment-matched evaluation.

```sh
python -m pip install -r requirements.txt
python -m examples.ch28_tabular.run_examples --output output/ch28.json
```

[Chapter 29: Scientific Sequences and Biological Structure](examples/ch29_protein/)
counts residues and ordered pairs, extracts contextual vectors with
ESMC-300M, and compares ESMFold2-Fast structures with experimental
references. The [chapter guide](examples/ch29_protein/README.md) includes
the protein records, evaluation rules, and figure commands.

The [11 exercises](examples/ch29_protein/EXERCISES.md) and
[complete solutions](solutions/ch29_protein/README.md) distinguish sequence
statistics, contextual pooling, contact geometry and structural confidence.

```sh
python -m pip install -r requirements.txt
python -m examples.ch29_protein.run_examples --output output/ch29.json
```

[Chapter 30: Decisions and Embodied Action](examples/ch30_decisions/)
builds return-to-go targets, fits two count policies, and runs them in a
small grid. The [chapter guide](examples/ch30_decisions/README.md)
explains the trajectory format and how the shifted trap affects a rollout.
It also reproduces the Vintix II decision-history comparison in Reacher
and the SmolVLA control-horizon comparison in LIBERO. The recorded
rollouts and simulator frames can redraw all three measured figures.

The [11 exercises](examples/ch30_decisions/EXERCISES.md) and
[complete solutions](solutions/ch30_decisions/README.md) connect causal action
prediction, return requests, changed environments and control timing.

```sh
python -m examples.ch30_decisions.run_examples --output output/ch30.json
```

[Chapter 31: What Changed, and What Remains?](examples/ch31_perspective/)
counts causal attention pairs and compares four ways to retain a
stream of delivery events. The [chapter guide](examples/ch31_perspective/README.md)
explains the two questions and the units used to record stored information.
It also shows how to reproduce the Qwen3.5-0.8B-Base measurements of
history tensors and paired choices when the prompt evidence changes.

The [12 exercises](examples/ch31_perspective/EXERCISES.md) and
[complete solutions](solutions/ch31_perspective/README.md) connect memory
accounting, information loss and controlled tests of retrieval and tracking.

```sh
python -m examples.ch31_perspective.run_examples --output output/ch31.json
```

## Examples for the appendices

The reference appendices use calculations from the chapter examples.

The book appendices are worked introductions without end-of-appendix exercises.
The earlier practice sets are retained here as optional companion material,
separate from the manuscript. Each has eight questions, complete written answers,
and an executed notebook. All six notebooks run offline on CPU using the canonical
chapter helpers. For new observations of pretrained models, see
[attention, loss gradients, and learned filters](examples/appendix_models/).

| Appendix | Exercises | Written answers | Executable answers |
| --- | --- | --- | --- |
| A. Notation and tensor shapes | [8 exercises](solutions/appendix_a_shapes/EXERCISES.md) | [PDF](solutions/appendix_a_shapes/written-solutions.pdf) | [Notebook](solutions/appendix_a_shapes/python-solutions.ipynb) / [PDF](solutions/appendix_a_shapes/python-solutions.pdf) |
| B. Calculus and optimization | [8 exercises](solutions/appendix_b_calculus/EXERCISES.md) | [PDF](solutions/appendix_b_calculus/written-solutions.pdf) | [Notebook](solutions/appendix_b_calculus/python-solutions.ipynb) / [PDF](solutions/appendix_b_calculus/python-solutions.pdf) |
| C. Stable numerical computation | [8 exercises](solutions/appendix_c_numerics/EXERCISES.md) | [PDF](solutions/appendix_c_numerics/written-solutions.pdf) | [Notebook](solutions/appendix_c_numerics/python-solutions.ipynb) / [PDF](solutions/appendix_c_numerics/python-solutions.pdf) |
| D. Complexity and memory accounting | [8 exercises](solutions/appendix_d_memory/EXERCISES.md) | [PDF](solutions/appendix_d_memory/written-solutions.pdf) | [Notebook](solutions/appendix_d_memory/python-solutions.ipynb) / [PDF](solutions/appendix_d_memory/python-solutions.pdf) |
| E. Convolution and parallel scans | [8 exercises](solutions/appendix_e_scans/EXERCISES.md) | [PDF](solutions/appendix_e_scans/written-solutions.pdf) | [Notebook](solutions/appendix_e_scans/python-solutions.ipynb) / [PDF](solutions/appendix_e_scans/python-solutions.pdf) |
| F. Diffusion foundations | [8 exercises](solutions/appendix_f_diffusion/EXERCISES.md) | [PDF](solutions/appendix_f_diffusion/written-solutions.pdf) | [Notebook](solutions/appendix_f_diffusion/python-solutions.ipynb) / [PDF](solutions/appendix_f_diffusion/python-solutions.pdf) |


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
