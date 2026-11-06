# Chapter 28. Learning from Tables

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch28_tabular.run_examples --output output/ch28.json
python -m unittest discover -s tests -p test_tabular.py -v
```

## Fit a row encoder on training data

`fit_row_encoder` learns a median from observed training numbers
and a sorted vocabulary of observed training categories. The
numeric column uses `None` and `NaN` for missing values. The category
column uses `None` for missing values and strings for observed ones.
The fit requires at least one observed numeric value.

`encode_rows` applies the fitted values to a new table with the
same two input columns. Output order is the filled numeric value,
its missing flag, one column per learned level, the categorical
missing flag, and the categorical unknown flag. A new observed
category activates the unknown flag. The first held-out row in
the example is missing its number and contains the new category
`kiosk`.

## Weight labels and check a probability score

`context_probability` applies softmax to supplied similarities
and returns a weighted mean of the context labels along with
the weights. The inputs are chosen to show the arithmetic of a
toy context rule. TabPFN obtains predictions from a pretrained
model with a learned inference procedure.

`brier_score` averages squared differences between binary labels
and positive-class probabilities. The worked probability example
uses a separate fixed prediction vector for this calculation.
