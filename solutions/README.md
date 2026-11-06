# Exercise solutions

## Chapter 1. Tokens, Embeddings, and Neural Language Models

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–12 | All written explanations and answers, from token IDs and shapes through probability, loss, context, and numerical stability | [Written solutions (PDF)](ch01_language/written-solutions.pdf) |
| 1–12 | Executable checks for every exercise, with full outputs | [Jupyter notebook](ch01_language/python-solutions.ipynb) · [Notebook export (PDF)](ch01_language/python-solutions.pdf) |

The [exercise statements](../examples/ch01_language/EXERCISES.md) match the revised
manuscript. Exercises 1–7 introduce the foundations, 8–10 combine calculations and
Python, and 11–12 are challenges. Both PDFs are generated from the same executed
notebook; [rebuild instructions](ch01_language/README.md) keep them synchronized.

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

All [14 exercises](../examples/ch02_recurrence/EXERCISES.md) have matching
numbering in the chapter, written solutions, and executed notebook.

| Coverage | Materials |
| --- | --- |
| Exercises 1-14: explanations, calculations, and interpretation | [Written solutions (PDF)](ch02_recurrence/written-solutions.pdf) |
| Exercises 1-14: executable checks and a complete color-reversal run | [Jupyter notebook](ch02_recurrence/python-solutions.ipynb) · [Notebook export (PDF)](ch02_recurrence/python-solutions.pdf) |

The [solution guide](ch02_recurrence/README.md) explains how to execute the
notebook and rebuild both PDFs. Install the [Chapter 2 requirements](../requirements-ch02.txt)
and a Jupyter kernel first. The 750-update experiment runs on a CPU in a
temporary directory, preserving existing checkpoints.

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

## Chapter 11. Long Context, Sparse Access, and Sequence Compression

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4, 10, and 12 | Attention grids, global relays, ranking and weighting, pooling ambiguity, access counts, and a long-document comparison | [Written solutions (PDF)](ch11_long_context/written-solutions.pdf) |
| 5–9 and 11 | Local and global masks, causal selection, summary windows, per-coordinate gates, summary visibility, and missed evidence | [Jupyter notebook](ch11_long_context/python-solutions.ipynb) · [Notebook export (PDF)](ch11_long_context/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[NumPy requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch11_long_context/python-solutions.ipynb`, select that Python
kernel, and run all cells. The notebook uses the
[Chapter 11 examples](../examples/ch11_long_context/) on CPU with fixed
synthetic arrays. The PDF export includes the code and its recorded outputs.
The two PDFs cover all four optional exercises.

## Chapter 12. Efficient Execution

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 10–11 | Response timing, batching, score storage, online softmax rescaling, and performance comparisons | [Written solutions (PDF)](ch12_efficient/written-solutions.pdf) |
| 5–9 | Uneven tile grids, tile-size agreement, empty tiles, causal offsets, and numerical stability | [Jupyter notebook](ch12_efficient/python-solutions.ipynb) · [Notebook export (PDF)](ch12_efficient/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[NumPy requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch12_efficient/python-solutions.ipynb`, select that Python
kernel, and run all cells. The notebook uses the
[Chapter 12 examples](../examples/ch12_efficient/) on CPU with fixed arrays.
The PDF export includes the code and its recorded outputs. The two PDFs
cover all three optional exercises.

## Chapter 13. Structured State Spaces and Long Convolutions

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 10–11 | State updates, filter lags, bilinear discretization, streaming storage, and time-varying coefficients | [Written solutions (PDF)](ch13_state_spaces/written-solutions.pdf) |
| 5–9 | Scan and convolution equivalence, circular wraparound, saved state, gate placement, and initial-state correction | [Jupyter notebook](ch13_state_spaces/python-solutions.ipynb) · [Notebook export (PDF)](ch13_state_spaces/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch13_state_spaces/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 13 examples](../examples/ch13_state_spaces/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 14. Selective State Spaces

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 10 | Selective updates, hold coefficients, affine composition, causal matrix entries, and memory tests | [Written solutions (PDF)](ch14_selective_ssm/written-solutions.pdf) |
| 5–9 | Stable small steps, per-coordinate retention, chunk equivalence, causality, and state reset | [Jupyter notebook](ch14_selective_ssm/python-solutions.ipynb) · [Notebook export (PDF)](ch14_selective_ssm/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch14_selective_ssm/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 14 examples](../examples/ch14_selective_ssm/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 15. Delta Rules and Hybrid Stacks

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–10 | Delta corrections, sign changes, Jamba layer types, mixed history bytes, overlapping keys, and controlled tests | [Written solutions (PDF)](ch15_hybrids/written-solutions.pdf) |
| 5–8 | State reads, persistent stores, mixer replacement, and cached causal attention | [Jupyter notebook](ch15_hybrids/python-solutions.ipynb) · [Notebook export (PDF)](ch15_hybrids/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch15_hybrids/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 15 examples](../examples/ch15_hybrids/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 16. From Pretrained Weights to Task Behavior

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 10–11 | Head parameters, shifted targets, low-rank updates, relative preferences, evaluation, and checkpoint reconstruction | [Written solutions (PDF)](ch16_adaptation/written-solutions.pdf) |
| 5–9 | Response masks, loss perturbations, adapter initialization, training updates, and detached reference scores | [Jupyter notebook](ch16_adaptation/python-solutions.ipynb) · [Notebook export (PDF)](ch16_adaptation/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements-ch05.txt).

```sh
python -m pip install -r requirements-ch05.txt jupyterlab
jupyter lab
```

Open `solutions/ch16_adaptation/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 16 examples](../examples/ch16_adaptation/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 17. Designing Reproducible Evaluations

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Scoring units, error costs, calibration, paired resampling, and reproducible comparison designs | [Written solutions (PDF)](ch17_evaluation/written-solutions.pdf) |
| 5–8 | Group splits, empty bins, paired intervals, and joining predictions by ID | [Jupyter notebook](ch17_evaluation/python-solutions.ipynb) · [Notebook export (PDF)](ch17_evaluation/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch17_evaluation/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 17 examples](../examples/ch17_evaluation/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 18. Language Understanding

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Word counts, masking, IOB2 alignment, span scores, label shifts, and evaluation design | [Written solutions (PDF)](ch18_understanding/written-solutions.pdf) |
| 5–8 | Masked pooling, first-piece labels, tag repair, and exact span metrics | [Jupyter notebook](ch18_understanding/python-solutions.ipynb) · [Notebook export (PDF)](ch18_understanding/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch18_understanding/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 18 examples](../examples/ch18_understanding/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 19. Translation and Summarization

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Fact preservation, shifted targets, path probabilities, overlap, source support, truncation, and adaptation evaluation | [Written solutions (PDF)](ch19_generation/written-solutions.pdf) |
| 5–8 | Padded batches, stopping and beam width, and directional overlap measures | [Jupyter notebook](ch19_generation/python-solutions.ipynb) · [Notebook export (PDF)](ch19_generation/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch19_generation/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 19 examples](../examples/ch19_generation/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 20. Search, Retrieval, and Grounded Assistants

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Term rarity, vector ranking, passage budgets, evidence recall, passage boundaries, abstention, and notice versions | [Written solutions (PDF)](ch20_retrieval/written-solutions.pdf) |
| 5–8 | BM25 length adjustment, candidate reranking, context packing, and literal matches | [Jupyter notebook](ch20_retrieval/python-solutions.ipynb) · [Notebook export (PDF)](ch20_retrieval/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch20_retrieval/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 20 examples](../examples/ch20_retrieval/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 21. Image and Video Understanding

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–12 | Patch grids, input scaling, attention counts, tubelets, and evaluation by recording, frame order, and resolution | [Written solutions (PDF)](ch21_vision/written-solutions.pdf) |
| 5–8 | Patch order and position, linear probes on ordered and sorted features, and divided space–time attention | [Jupyter notebook](ch21_vision/python-solutions.ipynb) · [Notebook export (PDF)](ch21_vision/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch21_vision/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 21 examples](../examples/ch21_vision/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 22. Object Detection and Segmentation

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Box and mask overlap, one-to-one assignment, ranked precision and recall, prompt protocols, coordinates, and error analysis | [Written solutions (PDF)](ch22_detection/written-solutions.pdf) |
| 5–8 | Touching edges, empty foreground, matching indices, overlap thresholds, and all-points AP | [Jupyter notebook](ch22_detection/python-solutions.ipynb) · [Notebook export (PDF)](ch22_detection/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch22_detection/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 22 examples](../examples/ch22_detection/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 23. Vision–Language and Multimodal Models

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Normalized similarities, directional probabilities, caption recall, visual projection, scene splits, and image-grounded evaluation | [Written solutions (PDF)](ch23_multimodal/written-solutions.pdf) |
| 5–8 | Logit scale, symmetric contrastive loss, rectangular retrieval, and visual cross-attention | [Jupyter notebook](ch23_multimodal/python-solutions.ipynb) · [Notebook export (PDF)](ch23_multimodal/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch23_multimodal/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 23 examples](../examples/ch23_multimodal/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 24. Speech and Audio

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Framing, frequency bins, decoding tasks, word alignments, speaker splits, false words, and streaming latency | [Written solutions (PDF)](ch24_speech_audio/written-solutions.pdf) |
| 5–8 | Spectrogram axes, tone peaks, pooled word error rate, and alignment ties | [Jupyter notebook](ch24_speech_audio/python-solutions.ipynb) · [Notebook export (PDF)](ch24_speech_audio/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch24_speech_audio/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 24 examples](../examples/ch24_speech_audio/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 25. Image and Video Generation

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Noise mixing, DDIM error, guidance, video attention, sampling latency, motion, and diversity | [Written solutions (PDF)](ch25_visual_generation/written-solutions.pdf) |
| 5–8 | Noise endpoints, exact and perturbed sampler paths, and guidance scales | [Jupyter notebook](ch25_visual_generation/python-solutions.ipynb) · [Notebook export (PDF)](ch25_visual_generation/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch25_visual_generation/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 25 examples](../examples/ch25_visual_generation/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 26. Models for Code

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Half-open intervals, code infilling, test boundaries, pass@k, repository context, and benchmark splits | [Written solutions (PDF)](ch26_code_models/written-solutions.pdf) |
| 5–8 | Per-case records, one-integer overlaps, pass@k, and shortest-candidate selection | [Jupyter notebook](ch26_code_models/python-solutions.ipynb) · [Notebook export (PDF)](ch26_code_models/python-solutions.pdf) |

The exercises use the Python standard library. Start JupyterLab from
the repository root.

```sh
python -m pip install jupyterlab
jupyter lab
```

Open `solutions/ch26_code_models/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 26 examples](../examples/ch26_code_models/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 27. Time-Series Forecasting

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Available information, split boundaries, patches, coverage, feature leakage, refits, and demand shifts | [Written solutions (PDF)](ch27_time_series/written-solutions.pdf) |
| 5–8 | Rolling windows, seasonal forecasts, drift, horizon errors, and interval widths | [Jupyter notebook](ch27_time_series/python-solutions.ipynb) · [Notebook export (PDF)](ch27_time_series/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch27_time_series/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 27 examples](../examples/ch27_time_series/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 28. Learning from Tables

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Available features, row encodings, weighted predictions, Brier score, customer splits, learning curves, and calibration | [Written solutions (PDF)](ch28_tabular/written-solutions.pdf) |
| 5–8 | Training-only fitting, category flags, context weights, and confidence scores | [Jupyter notebook](ch28_tabular/python-solutions.ipynb) · [Notebook export (PDF)](ch28_tabular/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch28_tabular/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 28 examples](../examples/ch28_tabular/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 29. Scientific Sequences and Biological Structure

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Alignment counts, dipeptides, pooling, geometric thresholds, family splits, embedding comparisons, and structural confidence | [Written solutions (PDF)](ch29_protein/written-solutions.pdf) |
| 5–8 | Normalized sequence features, padding, contact maps, and macro-F1 | [Jupyter notebook](ch29_protein/python-solutions.ipynb) · [Notebook export (PDF)](ch29_protein/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch29_protein/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 29 examples](../examples/ch29_protein/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 30. Decisions and Embodied Action

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–11 | Returns, context windows, action bins, episode splits, closed-loop outcomes, and control timing | [Written solutions (PDF)](ch30_decisions/written-solutions.pdf) |
| 5–8 | Returns-to-go, unsupported requests, layout shifts, and demonstration counts | [Jupyter notebook](ch30_decisions/python-solutions.ipynb) · [Notebook export (PDF)](ch30_decisions/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with the
[companion requirements](../requirements.txt).

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch30_decisions/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 30 examples](../examples/ch30_decisions/). The PDF export includes
the explanations, code, and recorded outputs.

## Chapter 31. What Changed, and What Remains?

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–4 and 9–12 | Causal pairs, architectural changes, summary fields, history budgets, recall tests, update composition, and ablation | [Written solutions (PDF)](ch31_perspective/written-solutions.pdf) |
| 5–8 | Window boundaries, missing events, and storage records for four memory rules | [Jupyter notebook](ch31_perspective/python-solutions.ipynb) · [Notebook export (PDF)](ch31_perspective/python-solutions.pdf) |

The exercises use the Python standard library. Start JupyterLab from
the repository root.

```sh
python -m pip install jupyterlab
jupyter lab
```

Open `solutions/ch31_perspective/python-solutions.ipynb`, select that Python kernel,
and run all cells. The notebook uses the
[Chapter 31 examples](../examples/ch31_perspective/). The PDF export includes
the explanations, code, and recorded outputs.
