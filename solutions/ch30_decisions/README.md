# Chapter 30: complete worked solutions

All [11 exercises](../../examples/ch30_decisions/EXERCISES.md) appear in:

- [Written solutions](written-solutions.pdf): explanations, calculations, tables and three figures.
- [Executed notebook](python-solutions.ipynb): all solutions with 21 executed code cells.
- [Notebook PDF](python-solutions.pdf): prose, code, figures and saved outputs.

The figures show causal action context, rewards after a layout change, and
the distinction between prediction and execution horizons. The examples use
count tables and a deterministic grid; they are not robot benchmark results.
The hypothetical episode groups and timing schedules are explicitly labeled.

## Run and reproduce

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch30_decisions/python-solutions.ipynb` and run all cells in order.
The saved run used Python 3.12.14, NumPy 2.5.4 and Matplotlib 3.11.2.
No model, external dataset, simulator service or robot is required.

Install Tectonic or XeLaTeX and export both PDFs from the executed notebook:

```sh
python -m pip install mistune
python solutions/ch30_decisions/build_pdfs.py
```

Use `--engine` for an explicit executable path. The exporter requires eleven
ordered solution headings, sequential execution and a result for every exercise.

## Canonical checks

```sh
python -m unittest discover -s tests -p test_decisions.py -v
python scripts/export_snippets.py --chapter ch30 --check
python -m examples.ch30_decisions.run_examples --output output/ch30.json
```

Ten tests pass and three canonical snippets match the manuscript.

## Primary background

- [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [Return conditioning and stochastic outcomes](https://arxiv.org/abs/2205.15967)
- [DAgger](https://proceedings.mlr.press/v15/ross11a.html)
- [RT-2](https://robotics-transformer2.github.io/assets/rt2.pdf)
- [OpenVLA](https://openvla.github.io/)
- [Physical Intelligence: pi0](https://www.pi.website/blog/pi0)
