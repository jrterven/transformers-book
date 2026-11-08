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

All [eleven exercises](../examples/ch25_visual_generation/EXERCISES.md) have
complete, ordered solutions in the [written PDF](ch25_visual_generation/written-solutions.pdf),
[executed notebook](ch25_visual_generation/python-solutions.ipynb), and
[notebook PDF](ch25_visual_generation/python-solutions.pdf).

Fifteen executed cells check noise mixtures, DDIM outputs and errors,
guidance increments, video pair counts and repeated compositions.
Three plots make the numerical behavior visible.
See the [solution guide](ch25_visual_generation/README.md) for setup and export.

## Chapter 26. Models for Code

All [eleven exercises](../examples/ch26_code_models/EXERCISES.md) have complete,
ordered solutions in the [written PDF](ch26_code_models/written-solutions.pdf),
[executed notebook](ch26_code_models/python-solutions.ipynb), and
[notebook PDF](ch26_code_models/python-solutions.pdf).

Seventeen executed cells check interval boundaries, infilling format,
per-case and complete-candidate scores, subset coverage, task aggregation,
selection, repository helper contracts, regressions and split rules.
Three plots show the key distinctions. See the [solution guide](ch26_code_models/README.md).

## Chapter 27. Time-Series Forecasting

All [eleven exercises](../examples/ch27_time_series/EXERCISES.md) have complete,
ordered solutions in the [written PDF](ch27_time_series/written-solutions.pdf),
[executed notebook](ch27_time_series/python-solutions.ipynb), and
[notebook PDF](ch27_time_series/python-solutions.pdf).

Eighteen executed cells check availability, split boundaries, patches, quantile
loss, windows, seasonal cycles, fixed drift, interval widths, leakage, weekly
refits and regime changes. Three plots explain the key distinctions.
See the [solution guide](ch27_time_series/README.md).

## Chapter 28. Learning from Tables

All [eleven exercises](../examples/ch28_tabular/EXERCISES.md) have complete,
ordered solutions in the [written PDF](ch28_tabular/written-solutions.pdf),
[executed notebook](ch28_tabular/python-solutions.ipynb), and
[notebook PDF](ch28_tabular/python-solutions.pdf).

Twenty executed cells check availability, fixed encodings, context probabilities,
confidence, empirical calibration, split units, nested data budgets and mature
outcomes. Three plots explain expected loss, normalized influence and calibration
versus risk separation. See the [solution guide](ch28_tabular/README.md).

## Chapter 29. Scientific Sequences and Biological Structure

All [eleven exercises](../examples/ch29_protein/EXERCISES.md) have complete,
ordered solutions in the [written PDF](ch29_protein/written-solutions.pdf),
[executed notebook](ch29_protein/python-solutions.ipynb), and
[notebook PDF](ch29_protein/python-solutions.pdf).

Eighteen executed cells check alignment counts, composition, dipeptide collisions,
masked pooling, translations and rotations, threshold boundaries, macro-F1,
family assignment, length eligibility and directional PAE. Three plots explain
the core distinctions. See the [solution guide](ch29_protein/README.md).

## Chapter 30. Decisions and Embodied Action

All [eleven exercises](../examples/ch30_decisions/EXERCISES.md) have full,
ordered solutions in the [written PDF](ch30_decisions/written-solutions.pdf),
[executed notebook](ch30_decisions/python-solutions.ipynb), and
[notebook PDF](ch30_decisions/python-solutions.pdf).

Twenty-one executed cells check reward indexing, discounted and undiscounted
returns, causal windows, action bins, absent targets, shifted rewards, tied
demonstrations, episode groups and refresh schedules. Three plots support the
explanations. See the [solution guide](ch30_decisions/README.md).

## Chapter 31. What Changed, and What Remains?

All [twelve exercises](../examples/ch31_perspective/EXERCISES.md) have full,
ordered solutions in the [written PDF](ch31_perspective/written-solutions.pdf),
[executed notebook](ch31_perspective/python-solutions.ipynb), and
[notebook PDF](ch31_perspective/python-solutions.pdf).

Nineteen executed cells check causal pairs, KV bytes, exact-summary capacity,
hybrid budgets, window boundaries, zero/missing values, storage units,
controlled histories, composed retrieval and forecast availability.
Three plots support the explanations. See the [solution guide](ch31_perspective/README.md).

## Optional practice for Appendices A-F

The book appendices are worked introductions without end-of-appendix exercises.
The earlier practice sets are retained here as optional companion material,
separate from the manuscript. Each has eight questions, complete written answers,
and an executed notebook. All six notebooks run offline on CPU using the canonical
chapter helpers. For new observations of pretrained models, see
[attention, loss gradients, and learned filters](../examples/appendix_models/).

| Appendix | Exercises | Written answers | Executable answers |
| --- | --- | --- | --- |
| A. Notation and tensor shapes | [8 exercises](appendix_a_shapes/EXERCISES.md) | [PDF](appendix_a_shapes/written-solutions.pdf) | [Notebook](appendix_a_shapes/python-solutions.ipynb) / [PDF](appendix_a_shapes/python-solutions.pdf) |
| B. Calculus and optimization | [8 exercises](appendix_b_calculus/EXERCISES.md) | [PDF](appendix_b_calculus/written-solutions.pdf) | [Notebook](appendix_b_calculus/python-solutions.ipynb) / [PDF](appendix_b_calculus/python-solutions.pdf) |
| C. Stable numerical computation | [8 exercises](appendix_c_numerics/EXERCISES.md) | [PDF](appendix_c_numerics/written-solutions.pdf) | [Notebook](appendix_c_numerics/python-solutions.ipynb) / [PDF](appendix_c_numerics/python-solutions.pdf) |
| D. Complexity and memory accounting | [8 exercises](appendix_d_memory/EXERCISES.md) | [PDF](appendix_d_memory/written-solutions.pdf) | [Notebook](appendix_d_memory/python-solutions.ipynb) / [PDF](appendix_d_memory/python-solutions.pdf) |
| E. Convolution and parallel scans | [8 exercises](appendix_e_scans/EXERCISES.md) | [PDF](appendix_e_scans/written-solutions.pdf) | [Notebook](appendix_e_scans/python-solutions.ipynb) / [PDF](appendix_e_scans/python-solutions.pdf) |
| F. Diffusion foundations | [8 exercises](appendix_f_diffusion/EXERCISES.md) | [PDF](appendix_f_diffusion/written-solutions.pdf) | [Notebook](appendix_f_diffusion/python-solutions.ipynb) / [PDF](appendix_f_diffusion/python-solutions.pdf) |

Rebuild an appendix's two PDFs from its executed notebook with
`python scripts/build_appendix_pdfs.py solutions/appendix_a_shapes`
(or the corresponding appendix directory). The exporter checks all eight
solutions and ordered execution before producing the PDFs.
