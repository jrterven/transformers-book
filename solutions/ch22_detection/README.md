# Chapter 22: complete worked solutions

All [11 exercises](../../examples/ch22_detection/EXERCISES.md) have complete
explanations and intermediate calculations in each edition.

- [Written solutions](written-solutions.pdf): explanations, tables and three plots.
- [Executed notebook](python-solutions.ipynb): 16 executed cells with saved outputs.
- [Notebook PDF](python-solutions.pdf): all prose, code, figures and results.

The plots show mask intersection and union, precision envelopes at two
IoU thresholds, and four detection errors. The solutions distinguish
one-to-one training assignment from ranked evaluation and identify both
optimal assignments in Exercise 3. These are teaching fixtures, not
benchmarks of pretrained detectors or segmentation models.

## Run the notebook

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch22_detection/python-solutions.ipynb`, restart its kernel
and run all cells in order on CPU. No downloads are needed. The saved run
uses Python 3.12.14, NumPy 2.5.4 and Matplotlib 3.11.2.

## Rebuild both PDFs

Install `mistune` and Tectonic or XeLaTeX, then run:

```sh
python -m pip install mistune
python solutions/ch22_detection/build_pdfs.py
```

Pass `--engine /path/to/tectonic` if needed. The exporter verifies all eleven
solution headings, sequential execution, a recorded result per exercise,
and absence of notebook errors and significant typesetting warnings.
Both PDFs come from this one executed notebook. Plot cells allow a page
break between their code and figure so tall plots remain inside the page.

## Verify the examples

```sh
python -m unittest discover -s tests -p test_detection.py -v
python scripts/export_snippets.py --chapter ch22 --check
python -m examples.ch22_detection.run_examples --output output/ch22.json
```

All ten chapter tests pass and four snippets match the manuscript.
Canonical helpers are unchanged. All-points AP is not the full COCO
protocol; empty mask union returns `None`. Returning masks to the original
image requires undoing padding and scale in the correct order.
