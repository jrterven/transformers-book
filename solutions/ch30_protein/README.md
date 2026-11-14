# Chapter 30: complete worked solutions

All [11 exercises](../../examples/ch30_protein/EXERCISES.md) are solved
step by step in each edition:

- [Written solutions](written-solutions.pdf): explanations, calculations, tables and three plots.
- [Executed notebook](python-solutions.ipynb): complete prose with 18 executed code cells.
- [Notebook PDF](python-solutions.pdf): prose, code, figures and saved results.

The plots show information lost by dipeptide counts, contact-map invariance and
cutoff sensitivity, and local versus cross-domain structural confidence.
The supplied vectors, confidence arrays and evaluation fixtures are teaching
examples, not outputs from pretrained protein models or experimental results.

## Run and reproduce

From the repository root:

```sh
python -m pip install -r requirements.txt matplotlib jupyterlab
jupyter lab
```

Open `solutions/ch30_protein/python-solutions.ipynb`, restart the kernel
and run all cells. The saved run used Python 3.12.14, NumPy 2.5.4 and
Matplotlib 3.11.2. No models or biological datasets are downloaded.

To export both PDFs, install Tectonic or XeLaTeX:

```sh
python -m pip install mistune
python solutions/ch30_protein/build_pdfs.py
```

Use `--engine` for an explicit executable path. The exporter checks eleven
ordered solution headings, sequential execution, exercise results and errors.

## Canonical checks

```sh
python -m unittest discover -s tests -p test_protein.py -v
python scripts/export_snippets.py --chapter ch30 --check
python -m examples.ch30_protein.run_examples --output output/ch30.json
```

All ten tests pass and four canonical snippets match the manuscript.
The sequence, pooling and geometry helpers are unchanged.

## Primary background

- [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)
- [Protein language representations](https://pmc.ncbi.nlm.nih.gov/articles/PMC8053943/)
- [GraphPart and relatedness-aware evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10578201/)
- [EMBL-EBI: local confidence](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/plddt-understanding-local-confidence/)
- [EMBL-EBI: predicted aligned error](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/pae-a-measure-of-global-confidence-in-alphafold-predictions/)
