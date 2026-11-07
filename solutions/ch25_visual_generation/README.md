# Chapter 25: complete worked solutions

All [11 exercises](../../examples/ch25_visual_generation/EXERCISES.md) are solved
step by step in each edition:

- [Written solutions](written-solutions.pdf): explanations, calculations and three plots.
- [Executed notebook](python-solutions.ipynb): every explanation with 15 executed cells.
- [Notebook PDF](python-solutions.pdf): prose, code, figures and saved results.

The plots show clean-estimate sensitivity, a fixed prediction error across
two sampler steps, and guidance as a straight line. Evaluation examples are
protocols or hypothetical records, not measurements of a trained generator.

## Run and reproduce

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch25_visual_generation/python-solutions.ipynb`, restart the
kernel and run all cells in order. The saved run used Python 3.12.14,
NumPy 2.5.4 and Matplotlib 3.11.2 on CPU. No model or dataset download is needed.

To rebuild both PDFs from that executed notebook, install Tectonic or
XeLaTeX and run:

```sh
python -m pip install mistune
python solutions/ch25_visual_generation/build_pdfs.py
```

Use `--engine` for an explicit engine path. The exporter checks all eleven
exercise headings, sequential execution, a recorded result for each exercise
and no execution errors.

## Canonical checks

```sh
python -m unittest discover -s tests -p test_visual_generation.py -v
python scripts/export_snippets.py --chapter ch25 --check
python -m examples.ch25_visual_generation.run_examples --output output/ch25.json
```

All eight tests pass and all three canonical listings match the manuscript.
The numerical helpers are unchanged. `add_noise` allows a zero signal
coefficient; `ddim_step` requires a strictly positive current coefficient.
Guidance extrapolates noise estimates, which are not probabilities.
