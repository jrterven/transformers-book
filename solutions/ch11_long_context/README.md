# Chapter 11 worked solutions

All twelve [exercise statements](../../examples/ch11_long_context/EXERCISES.md) are solved in each edition:

- [Written solutions](written-solutions.pdf): full explanations, calculations, and two figures.
- [Executed notebook](python-solutions.ipynb): the same solutions plus 19 code cells and saved results.
- [Notebook PDF](python-solutions.pdf): code, results, figures, and every explanation.

The progression starts with attention grids, causal global relays, ranking versus weighting, and pooling ambiguity. It then checks masks, summary windows, per-coordinate gates, causal visibility, joint softmax, storage/read counts, and the consequences of missing evidence. Exercise 12 gives a controlled long-document evaluation protocol; it does not claim measured benchmark results.

## Run and export

From the repository root:

```sh
python -m pip install -r requirements.txt jupyterlab matplotlib mistune
jupyter lab solutions/ch11_long_context/python-solutions.ipynb
```

Run all cells in order and save the notebook. To regenerate both PDFs from those saved results, install Tectonic or XeLaTeX and run:

```sh
python solutions/ch11_long_context/build_pdfs.py
```

Use `--engine /path/to/tectonic` or `--engine /path/to/xelatex` if needed. The exporter requires all exercises in order, sequential execution counts, recorded numerical results, and clean typesetting. It preserves the notebook plots in both editions.

These CPU examples use fixed synthetic inputs. Their causal rule permits a summary when its end index is at or before the query, assuming causal source features and gates. The DeepSeek-V4 report uses preceding blocks only and includes a learned sink; these teaching examples deliberately isolate the simpler operations.
