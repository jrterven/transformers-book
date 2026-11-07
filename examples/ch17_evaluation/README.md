# Chapter 17. Designing Reproducible Evaluations

Run the NumPy examples from the repository root.

```sh
python -m pip install -r requirements.txt
python -m examples.ch17_evaluation.run_examples --output output/ch17.json
python -m unittest discover -s tests -p test_evaluation.py -v
```

## Split groups before scoring rows

`split_by_group` shuffles the distinct source IDs with a fixed seed,
allocates the specified counts of groups to train, validation, and
test, and returns row indices for those three partitions. All rows
from a group stay together. Unequal group sizes can give different
proportions of rows and groups. The fixture uses six groups of sizes
2, 3, 1, 2, 3, and 1.

## Read the errors and the scores

`binary_metrics` expects binary labels and predictions in the same
row order. It returns the four confusion counts and accuracy,
precision, recall, specificity, and F1. A zero denominator gives
`None` for that metric. Both models in the fixture have 75% accuracy
and make different errors.

`reliability_bins` uses positive-class probabilities. Each interior
edge starts the next bin, and the last bin includes probability one.
Each nonempty bin reports its row count, mean probability, and
observed positive fraction. An empty bin has count zero and `NaN`
for its two means. The Brier score is the mean squared difference
between the probability and the binary label across all rows.
The calibration example uses its own fixture of 12 scored cases.

## Resample paired cases

Align both models' predictions to the same row IDs before calling
`paired_accuracy_bootstrap`. It calculates each row's difference in
correctness as B minus A. A replicate draws rows with replacement and
averages their differences. Passing `groups` instead draws complete
groups with replacement. Each grouped replicate divides the sum of
the sampled correctness differences by the number of sampled rows,
including repeated groups, so its statistic remains item-weighted.

The function returns the original accuracy gap and the percentile
interval of the resampled gaps. The sampling unit determines which
dependencies are preserved. The examples compare row and group
resampling on six small groups and use a fixed seed for reproducibility.


## Exercises and complete solutions

The [twelve exercises](EXERCISES.md) progress from calculations to code and
controlled evaluation designs. The [solutions guide](../../solutions/ch17_evaluation/README.md)
links the executed notebook and both PDF editions; each covers every exercise.

Choose the question before the denominator. Per-check accuracy, the mean
of conversation accuracies, and complete-conversation success can all differ.
For binary probabilities, the Brier score averages `(p-y)**2` for the named
positive class. A constant prevalence estimate can be empirically calibrated
while failing to distinguish cases. Error costs can make a threshold below
0.5 appropriate; tune it using validation data, then freeze it for testing.

Grouped resampling preserves within-group dependence. It does not automatically
change the helper's per-message statistic into an equal-customer average.
The nominal confidence level is an approximate repeated-sampling coverage
target. An interval spanning zero does not prove equivalence, and more draws
do not add independent evaluation cases. Training-seed variation is separate.
The helper stores a draw-index matrix for these small fixtures; large-scale
evaluation can process bootstrap draws in bounded batches instead.
