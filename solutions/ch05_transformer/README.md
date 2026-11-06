# Chapter 5 worked solutions

Juan Terven and Diana Córdova

All **15 exercises** in [EXERCISES.md](../../examples/ch05_transformer/EXERCISES.md)
and the manuscript have matching solutions:

- [Written solutions](written-solutions.pdf): reasoning, intermediate calculations, and interpretations for every exercise.
- [Executed notebook](python-solutions.ipynb): the same explanations, with runnable checks and recorded outputs.
- [Notebook PDF](python-solutions.pdf): explanations, code, and outputs in reading order.

Exercises 1–7 establish the information paths, axes, positions, normalization,
and dropout. Exercises 8–13 calculate and test complete components. Exercises
14–15 investigate a deliberately broken causal mask and independent deeper
stacks. The temporary mask experiment restores the original function when its
context ends; it does not edit the canonical model source.

## Run and verify

From the repository root, use a Python environment with the chapter dependencies:

```sh
python -m pip install -r requirements-ch05.txt jupyterlab
jupyter lab solutions/ch05_transformer/python-solutions.ipynb
```

Restart the kernel and run all cells in order. The checked notebook has 26
executed code cells, no execution errors, and a recorded result for every
exercise. It uses CPU float64; the validated environment was Python 3.12.14,
PyTorch 2.15.0, and NumPy 2.5.4. All random models are untrained.

```sh
python -m unittest discover -s tests -p test_transformer.py -v
python scripts/export_snippets.py --chapter ch05 --check
```

The 21 tests cover numerical calculations, masks, information-flow
interventions, dropout behavior, and gradients, including comparison of full
stacks with PyTorch's reference implementation. The nine manuscript excerpts
are exported from canonical source, not maintained separately.

## Rebuild both PDFs

The executed notebook is the single source for both PDF editions. Install
`mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch05_transformer/build_pdfs.py
```

Use `--engine /path/to/tectonic` when the engine is not on PATH. The exporter
checks ordered execution, complete exercise coverage, and recorded results.
It rejects execution errors, missing characters, and overfull typesetting
before replacing the PDFs. Inspect the rendered pages after substantive edits.
