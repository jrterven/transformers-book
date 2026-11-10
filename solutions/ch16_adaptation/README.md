# Chapter 16: complete worked solutions

All [twelve exercises](../../examples/ch16_adaptation/EXERCISES.md) are solved
in every edition:

- [Written PDF](written-solutions.pdf): every derivation, result and interpretation.
- [Executed notebook](python-solutions.ipynb): explanations, 26 code cells and two plots.
- [Notebook PDF](python-solutions.pdf): all code and saved outputs.

The sequence covers task-head counts, shifted targets, low-rank corrections,
relative preference gaps, token and response averages, zero-initialized
factors, gradients through frozen operations, checkpoint probes and a fair
comparison protocol. The plots compare loss reductions and show the DPO
loss with its gradient. These are mechanism checks, not benchmark results.

## Run and export

From the repository root:

```sh
python -m pip install -r requirements-ch05.txt matplotlib jupyterlab mistune
jupyter lab
```

Open this notebook, restart its kernel and run every cell. The saved CPU run
uses Python 3.12.14, PyTorch 2.15.0, NumPy 2.5.4 and Matplotlib 3.11.2.
No external model or dataset downloads are required. The fixed classifier
initialization and explicit seed for the fresh adapter make the examples
reproducible; all numerical checks use stated tolerances.

With Tectonic or XeLaTeX available, rebuild both PDF editions:

```sh
python solutions/ch16_adaptation/build_pdfs.py
```

Use `--engine /path/to/tectonic` if necessary. The exporter checks the twelve
exercise headings, saved results and sequential execution counts. Temporary
files go to `output/ch16-solutions-build/`. Rebuild both PDFs after edits.

## Canonical validation

```sh
python -m unittest discover -s tests -p test_adaptation.py -v
python -m examples.ch16_adaptation.run_examples --output output/ch16.json
python scripts/export_snippets.py --chapter ch16 --check
```

The 13 chapter tests pass, and all seven manuscript listings match their
canonical sources. The notebook adds independent checks for both batch
reductions, the first two scalar LoRA gradients, frozen gradient paths,
affine basis probes and same-shaped query/value adapters.

The optional [actual pretrained-model experiment](../../examples/ch16_adaptation/PRETRAINED_FIGURES.md)
is separate from these exercise solutions. Its real training checkpoints and
figures do not change the twelve exercise statements or their solutions.
