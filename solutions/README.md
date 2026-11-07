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

All [14 exercises](../examples/ch03_alignment/EXERCISES.md) have matching
numbering in the chapter, written solutions, and executed notebook.

| Coverage | Materials |
| --- | --- |
| Exercises 1–14: explanations, calculations, and interpretation | [Written solutions (PDF)](ch03_alignment/written-solutions.pdf) |
| Exercises 1–14: executable checks and a complete attention experiment | [Jupyter notebook](ch03_alignment/python-solutions.ipynb) · [Notebook export (PDF)](ch03_alignment/python-solutions.pdf) |

The [solution guide](ch03_alignment/README.md) explains how to execute the
notebook and rebuild both PDFs. Install the [Chapter 3 requirements](../requirements-ch03.txt)
and a Jupyter kernel first. The 750-update experiment runs on one CPU thread
in a temporary directory. Exercises move from weighted averages to source
masking, decoder timing, measured alignment maps, and an optional gradient challenge.

## Chapter 4. Self-Attention from Vectors to Matrices

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–15 | Roles, weighted reads, scaling, axes, masks, information flow, permutations, and optional gradients | [Written solutions (PDF)](ch04_attention/written-solutions.pdf) · [Jupyter notebook](ch04_attention/python-solutions.ipynb) · [Notebook export (PDF)](ch04_attention/python-solutions.pdf) |

Start JupyterLab from the repository root in an environment with NumPy.

```sh
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Open `solutions/ch04_attention/python-solutions.ipynb`, select that Python
kernel, and run all cells. The notebook uses the NumPy implementation from the
[Chapter 4 examples](../examples/ch04_attention/). The [exercise sheet](../examples/ch04_attention/EXERCISES.md)
and [solution guide](ch04_attention/README.md) describe the progression and how
to regenerate both PDFs from the executed notebook.

## Chapter 5. Multiple Heads, Position, and the Complete Architecture

All [15 exercises](../examples/ch05_transformer/EXERCISES.md) are covered in
each edition, in the same order as the chapter.

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–15 | Prediction paths, projected heads, positions, post-norm, dropout, FFNs, vocabulary loss, padding, causality, and deeper stacks | [Written solutions (PDF)](ch05_transformer/written-solutions.pdf) · [Jupyter notebook](ch05_transformer/python-solutions.ipynb) · [Notebook export (PDF)](ch05_transformer/python-solutions.pdf) |

See the [Chapter 5 solution guide](ch05_transformer/README.md) for execution,
validation, and reproducible PDF export instructions. The written PDF and
notebook PDF are generated from the same executed notebook.

## Chapter 6. Training and Generating with a Small Transformer

All [15 exercises](../examples/ch06_training/EXERCISES.md) are solved in each
edition, in the chapter's order.

| Exercises | Topics | Files |
| --- | --- | --- |
| 1–15 | Updates, shifts, loss, clipping, generation, caches, checkpoints, generalization, selection, leakage, and timing | [Written solutions (PDF)](ch06_training/written-solutions.pdf) · [Jupyter notebook](ch06_training/python-solutions.ipynb) · [Notebook export (PDF)](ch06_training/python-solutions.pdf) |

See the [solution guide](ch06_training/README.md) for execution and PDF export.
The notebook trains small CPU models, includes all four optional experiments,
and restores temporary mask changes automatically. Both PDF editions are
built from that executed notebook.

## Chapter 7. GPT, BERT, and T5

All [14 exercises](../examples/ch07_pretraining/EXERCISES.md) are covered in each edition.

| Edition | Contents | File |
| --- | --- | --- |
| Written | Step-by-step explanations, calculations, and measured result tables for all 14 exercises | [Written solutions (PDF)](ch07_pretraining/written-solutions.pdf) |
| Executable | The same explanations plus 23 executed Python cells and saved outputs | [Notebook](ch07_pretraining/python-solutions.ipynb) |
| Python PDF | The complete executed notebook typeset for reading and printing | [Notebook export (PDF)](ch07_pretraining/python-solutions.pdf) |

Exercises 1-7 build intuition; 8-11 investigate corruption, gradients,
visibility, and fitting the prepared examples; optional exercises 12-14
study encoder-decoder paths, classification heads, and fair comparison.
The last exercise specifies an evaluation protocol. The classifier compares
random starting weights and makes no claim about pretrained transfer.
See the [reproduction and export guide](ch07_pretraining/README.md).

## Chapter 8. From the Original Block to Modern Decoders

All [14 exercises](../examples/ch08_modern/EXERCISES.md) have complete solutions
in each format:

| Format | Contents |
| --- | --- |
| [Written solutions (PDF)](ch08_modern/written-solutions.pdf) | All explanations, derivations, and results |
| [Executed notebook](ch08_modern/python-solutions.ipynb) | The same solutions with 20 code cells and saved outputs |
| [Notebook export (PDF)](ch08_modern/python-solutions.pdf) | The full notebook typeset for reading and printing |

The progression covers pre-norm, RMSNorm geometry, SwiGLU, RoPE signs and
multiple frequencies, model checks, cache positions, and residual gradients.
The final exercise plans a matched architecture comparison. All examples run
locally on CPU. See the [reproduction and export guide](ch08_modern/README.md).

## Chapter 9. Scaling Data, Compute, and Expert Capacity

All [14 exercises](../examples/ch09_scaling/EXERCISES.md) have complete solutions
in each edition, ordered from hand calculations to Python investigations and
three optional extensions.

| Edition | Contents |
| --- | --- |
| [Written solutions (PDF)](ch09_scaling/written-solutions.pdf) | All explanations, calculations, tables, and the allocation plot |
| [Executed notebook](ch09_scaling/python-solutions.ipynb) | The same solutions with 23 code cells and saved outputs |
| [Notebook export (PDF)](ch09_scaling/python-solutions.pdf) | The complete notebook typeset for reading and printing |

The exercises compare dense allocations, corpus reuse, expert weight counts,
Switch and top-two gates, capacity, overflow, balance diagnostics, communication,
and the continuous optimum of the teaching loss. The final exercise designs a
controlled experiment. See the [reproduction and export guide](ch09_scaling/README.md).

## Chapter 10. Incremental Decoding and KV Representations

All [12 exercises](../examples/ch10_kv/EXERCISES.md) are solved in every edition:

| Edition | Contents |
| --- | --- |
| [Written solutions](ch10_kv/written-solutions.pdf) | Complete explanations, derivations, tables, and the chunk-mask diagram |
| [Executed notebook](ch10_kv/python-solutions.ipynb) | The same solutions with 19 executed code cells and recorded outputs |
| [Notebook PDF](ch10_kv/python-solutions.pdf) | Every explanation, code cell, diagram, and result |

The four hand calculations lead to four Python investigations and four
optional extensions. Topics include prediction timing, attention work,
shared KV heads, cache bytes, causal chunks, storage rounding, RoPE positions,
and MLA reassociation. See the [reproduction and export guide](ch10_kv/README.md).

## Chapter 11. Long Context, Sparse Access, and Sequence Compression

All twelve exercises are covered in the [written PDF](ch11_long_context/written-solutions.pdf), [executed notebook](ch11_long_context/python-solutions.ipynb), and [notebook PDF](ch11_long_context/python-solutions.pdf). The solutions include causal relays, selection and renormalization, ambiguous summaries, joint normalization, memory/read accounting, and a controlled evaluation protocol.

See the [exercise statements](../examples/ch11_long_context/EXERCISES.md) and [run/export instructions](ch11_long_context/README.md). The notebook contains 19 executed cells with saved results and two explanatory figures.

## Chapter 12. Efficient Execution

All **11 exercises** are fully solved in every edition: [written PDF](ch12_efficient/written-solutions.pdf), [executed notebook](ch12_efficient/python-solutions.ipynb), and [notebook PDF](ch12_efficient/python-solutions.pdf).

The sequence covers response timing, score storage, online softmax, partial tiles,
masked rows, query offsets, numerical stability, and fair performance measurement.
Sixteen executed code cells and two diagrams make the calculations inspectable.
See the [exercise statements](../examples/ch12_efficient/EXERCISES.md) and
[run/export guide](ch12_efficient/README.md). The written edition includes the
Python investigations' explanations and results as well as the conceptual answers.

## Chapter 13. Structured State Spaces and Long Convolutions

All **11 exercises** are fully solved in every edition: [written PDF](ch13_state_spaces/written-solutions.pdf), [executed notebook](ch13_state_spaces/python-solutions.ipynb), and [notebook PDF](ch13_state_spaces/python-solutions.pdf).

The sequence covers state summaries, lag filters, discretization, FFT padding,
chunk continuation, gate placement, saved-state corrections, streaming storage,
and time-varying updates. Seventeen executed code cells and two plots expose
the calculations. See the [exercise statements](../examples/ch13_state_spaces/EXERCISES.md)
and [run/export guide](ch13_state_spaces/README.md). The written edition also
includes the Python investigations' explanations and numerical results.

## Chapter 14. Selective State Spaces

All **10 exercises** are fully solved in every edition: [written PDF](ch14_selective_ssm/written-solutions.pdf), [executed notebook](ch14_selective_ssm/python-solutions.ipynb), and [notebook PDF](ch14_selective_ssm/python-solutions.pdf).

The sequence covers selective updates, sigmoid write gates, affine composition,
causal matrix entries, stable small steps, coordinate retentions, chunk handoffs,
resets, and memory-test design. Twenty-one executed code cells and two plots
make the calculations inspectable. See the [exercise statements](../examples/ch14_selective_ssm/EXERCISES.md)
and [run/export guide](ch14_selective_ssm/README.md).

## Chapter 15. Delta Rules and Hybrid Stacks

All **10 exercises** are completely solved in every edition: [written PDF](ch15_hybrids/written-solutions.pdf), [executed notebook](ch15_hybrids/python-solutions.ipynb), and [notebook PDF](ch15_hybrids/python-solutions.pdf).

Follow predictions and corrections, compare queries and overlapping writes,
trace separate stores through a hybrid, and test causal cache equivalence.
The notebook has 21 executed code cells and two plots. The final design varies
table size and update length separately and distinguishes circular counting
from order-sensitive swaps. See the [exercise statements](../examples/ch15_hybrids/EXERCISES.md)
and [run/export guide](ch15_hybrids/README.md).

## Chapter 16. From Pretrained Weights to Task Behavior

All twelve exercises are fully solved in each edition:

| Edition | Contents | File |
| --- | --- | --- |
| Written | Derivations, exact checks, two plots and comparison protocol | [PDF](ch16_adaptation/written-solutions.pdf) |
| Notebook | The same explanations with 26 executed code cells | [Jupyter notebook](ch16_adaptation/python-solutions.ipynb) |
| Notebook export | Every code cell and saved result | [PDF](ch16_adaptation/python-solutions.pdf) |

See the [exercise statements](../examples/ch16_adaptation/EXERCISES.md)
and [run/export guide](ch16_adaptation/README.md).

## Chapter 17. Designing Reproducible Evaluations

All twelve exercises are fully solved in each edition:

| Edition | Contents | File |
| --- | --- | --- |
| Written | Calculations, interpretations, two plots and comparison designs | [PDF](ch17_evaluation/written-solutions.pdf) |
| Notebook | The same explanations with 21 executed code cells | [Jupyter notebook](ch17_evaluation/python-solutions.ipynb) |
| Notebook export | Every code cell and saved result | [PDF](ch17_evaluation/python-solutions.pdf) |

See the [exercise statements](../examples/ch17_evaluation/EXERCISES.md)
and [run/export guide](ch17_evaluation/README.md).

## Chapter 18. Language Understanding

All twelve exercises are fully solved in each edition:

| Edition | Contents | File |
| --- | --- | --- |
| Written | Calculations, interpretations, two plots and evaluation designs | [PDF](ch18_understanding/written-solutions.pdf) |
| Notebook | The same explanations with 19 executed code cells | [Jupyter notebook](ch18_understanding/python-solutions.ipynb) |
| Notebook export | Every code cell and saved result | [PDF](ch18_understanding/python-solutions.pdf) |

See the [exercise statements](../examples/ch18_understanding/EXERCISES.md)
and [run/export guide](ch18_understanding/README.md).

## Chapter 19. Translation and Summarization

All twelve exercises have complete solutions in every edition, including
the new corpus-aggregation calculation and the source-support audit.

| Edition | Contents |
| --- | --- |
| [Written solutions](ch19_generation/written-solutions.pdf) | Explanations, calculations, tables, and two plots for Exercises 1-12 |
| [Executed notebook](ch19_generation/python-solutions.ipynb) | The same solutions with 18 executed cells and saved outputs |
| [Notebook PDF](ch19_generation/python-solutions.pdf) | All explanations, code, figures, and results |

See the [exercise statements](../examples/ch19_generation/EXERCISES.md)
and [run/export guide](ch19_generation/README.md).

## Chapter 20. Search, Retrieval, and Grounded Assistants

All twelve exercises have complete worked solutions, including the RAG
mixture calculation and the source-version audit.

| Edition | Contents |
| --- | --- |
| [Written solutions](ch20_retrieval/written-solutions.pdf) | Explanations, calculations, tables and two plots for Exercises 1-12 |
| [Executed notebook](ch20_retrieval/python-solutions.ipynb) | The same solutions with 17 executed code cells and saved results |
| [Notebook PDF](ch20_retrieval/python-solutions.pdf) | All explanations, code, figures and outputs |

See the [exercise statements](../examples/ch20_retrieval/EXERCISES.md)
and [run/export guide](ch20_retrieval/README.md).

## Chapter 21. Image and Video Understanding

All twelve exercises have complete worked solutions, including patch
ordering, position information and controlled visual evaluation.

| Edition | Contents |
| --- | --- |
| [Written solutions](ch21_vision/written-solutions.pdf) | Explanations, calculations, tables and two plots for Exercises 1-12 |
| [Executed notebook](ch21_vision/python-solutions.ipynb) | The same solutions with 19 executed cells and saved outputs |
| [Notebook PDF](ch21_vision/python-solutions.pdf) | All explanations, code, figures and results |

See the [exercise statements](../examples/ch21_vision/EXERCISES.md)
and [run/export guide](ch21_vision/README.md).

## Chapter 22. Object Detection and Segmentation

All eleven exercises have complete worked solutions, including both
minimum-cost assignments, precision envelopes and coordinate inversion.

| Edition | Contents |
| --- | --- |
| [Written solutions](ch22_detection/written-solutions.pdf) | Explanations, calculations, tables and three plots for Exercises 1-11 |
| [Executed notebook](ch22_detection/python-solutions.ipynb) | The same solutions with 16 executed cells and saved outputs |
| [Notebook PDF](ch22_detection/python-solutions.pdf) | All explanations, code, figures and results |

See the [exercise statements](../examples/ch22_detection/EXERCISES.md)
and [run/export guide](ch22_detection/README.md).

## Chapter 23. Vision-Language and Multimodal Models

All eleven exercises have complete explanations in all three editions.
They cover candidate-dependent probabilities, symmetric contrastive loss,
multiple-positive retrieval, visual projection and paired evaluation.

| Edition | Contents |
| --- | --- |
| [Written solutions](ch23_multimodal/written-solutions.pdf) | All eleven worked solutions, tables and three plots |
| [Executed notebook](ch23_multimodal/python-solutions.ipynb) | The same explanations with 14 executed cells and saved outputs |
| [Notebook PDF](ch23_multimodal/python-solutions.pdf) | All prose, code, figures and results |

See the [exercise statements](../examples/ch23_multimodal/EXERCISES.md)
and [run/export guide](ch23_multimodal/README.md).

## Chapter 24. Speech and Audio

All [eleven exercises](../examples/ch24_speech_audio/EXERCISES.md) have complete,
ordered solutions in the [written PDF](ch24_speech_audio/written-solutions.pdf),
[executed notebook](ch24_speech_audio/python-solutions.ipynb), and
[notebook PDF](ch24_speech_audio/python-solutions.pdf).

Sixteen executed cells check frame counts, frequency axes, transcript alignments,
pooled WER, no-speech insertions, speaker splits and caption latency. Three plots
show overlapping windows, tone spectra and first-correct versus stable text.
See the [chapter solution guide](ch24_speech_audio/README.md) for setup and export.

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
