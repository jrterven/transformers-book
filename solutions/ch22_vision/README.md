# Chapter 22: complete worked solutions

All [12 exercises](../../examples/ch22_vision/EXERCISES.md) have complete
explanations and intermediate calculations in each edition.

- [Written solutions](written-solutions.pdf): explanations, tables and two plots.
- [Executed notebook](python-solutions.ipynb): 19 executed cells with saved outputs.
- [Notebook PDF](python-solutions.pdf): all prose, code, figures and results.

The sequence moves from image shapes and scaling to patch extraction,
position information, linear probes, video attention and evaluation design.
The two plots trace pixels into patch rows and compare divided attention
counts. Synthetic features illustrate information retention; they are not
measurements of pretrained vision models.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch22_vision/python-solutions.ipynb`, restart its Python
kernel and run all cells in order on CPU. No model or dataset download is
needed. Saved outputs use Python 3.12.14, NumPy 2.5.4 and Matplotlib 3.11.2.

## Rebuild both PDFs

Install `mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch22_vision/build_pdfs.py
```

Pass `--engine /path/to/tectonic` if necessary. The exporter verifies all
twelve solution headings, sequential execution, a recorded result for
each exercise, and absence of notebook errors and significant typesetting
warnings. Both PDFs come from the same executed notebook.

## Verify the examples

```sh
python -m unittest discover -s tests -p test_vision.py -v
python scripts/export_snippets.py --chapter ch22 --check
python -m examples.ch22_vision.run_examples --output output/ch22.json
```

All nine chapter tests passed and four canonical snippets match the
manuscript. The helper implementation is unchanged. Flattening preserves
values; a narrow projection can discard information. Location requires
position information in a standard ViT, and temporal attention requires
order information to distinguish reversed actions.
