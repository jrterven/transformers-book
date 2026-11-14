# Chapter 29. Learning from Tables

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch29_tabular.run_examples --output output/ch29.json
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

## Interpret the representation and scores

With K known categories, the teaching encoder has K+4 columns. An imputed value
retains its missingness flag. A present but unseen category activates the
unknown column; two unseen names collapse to the same representation.
Reserving a column does not teach a model how a new category behaves.

The context-weighted label sum illustrates normalized influence and is not a
TabPFN implementation. Increasing a negative row's similarity can lower the
positive estimate. Brier loss measures overall probability error; a reliability
plot answers a different question. Two empirically calibrated predictors can
have different Brier scores when one separates risk groups better.

## Classify raisins and predict concrete strength

Both demos use released 2026 checkpoints for [TabICLv2](https://arxiv.org/abs/2602.11139)
and [Mitra-v2](https://arxiv.org/abs/2609.04540). In each task the pretrained
weights stay fixed. A constant, a scaled linear model, and histogram-based
gradient-boosted trees provide baselines. Each method gets the same 32,
128, or full set of labeled rows. The smaller sets are nested inside the
larger ones. Each task has a fixed test set, and no settings are selected
using its outcomes.

- **Raisin classification.** The seven numeric features describe the shape
  and size of each photographed raisin. Besni is class 1 and Kecimen is
  class 0. A stratified split puts 720 rows in training and 180 in test,
  with 90 test cases of each variety. At a threshold of 0.5, the two
  Transformers each classify 159 cases correctly. Accuracy, ROC AUC,
  Brier score, and reliability describe different properties of the predictions.
- **Concrete regression.** Curing age and seven ingredient amounts predict
  measured strength in MPa. After dropping 25 exact duplicate records,
  the split groups by the seven-ingredient vector, with 743 rows from 320
  recipes in training and 262 rows from 107 different recipes in test.
  Rows with identical features and different measured outcomes are
  retained. The groups are used only for the split. TabICLv2 and Mitra-v2
  obtain mean absolute errors of 3.14 and 3.19 MPa on these test rows.
  Their central 80% intervals cover 191 and 212 rows, respectively.

### Inspect predictions and redraw the figures

The [recorded predictions](records/tabular_runs.json) contain all models
and budgets. Within each dataset, `test_ids` gives the zero-based row
offsets in the original CSV (excluding the header). It has the same
order as `actual`, `point`, and any `quantiles`. The ordered
`training_ids` list gives the exact prefix used at each budget.
For Raisin, `point` is a Besni probability. For concrete, it is
a mean in MPa, and the optional quantile columns are at 0.1, 0.5, and
0.9. The recipe figure uses the test recipe with the most recorded ages,
breaking ties by the smallest group ID.

Use Python 3.12 for the demos. To plot the committed predictions, install
the plotting dependencies in a fresh environment.

```sh
python3.12 -m venv .venv-tabular
source .venv-tabular/bin/activate
python -m pip install "numpy==2.5.0" "matplotlib==3.11.2" "scikit-learn==1.9.1"
python -m examples.ch29_tabular.make_figures --output-dir output/ch29-figures
```

### Rerun both models

```sh
python -m pip install -r examples/ch29_tabular/requirements-pretrained.txt
python -m examples.ch29_tabular.run_pretrained --output output/ch29-replay.json
python -m examples.ch29_tabular.make_figures \
  --predictions output/ch29-replay.json --output-dir output/ch29-figures
```

The runner downloads about 830 MB of model weights on first use and
verifies the SHA-256 digests before loading them. It uses four CPU threads
by default. A local mirror can be passed with `--checkpoints PATH`.
Place the two TabICL files in `tabicl/` and each Mitra pair in
`mitra-classifier/` or `mitra-regressor/` within that directory.
The [pinned TabICL snapshot](https://huggingface.co/jingang/TabICL/tree/4dcd344ece2c00be9e831fdd35bed57b5ad83e19)
contains the two `v2-20260212.ckpt` files. The [Mitra classifier snapshot](https://huggingface.co/autogluon/mitra-classifier-2/tree/edada0d20759c58ada8c8605c25f22f6e98ea5f0)
and [Mitra regressor snapshot](https://huggingface.co/autogluon/mitra-regressor-2/tree/246c7d310fff04296a2388df5f448e3ffe809283)
each provide `config.json` and `model.safetensors`.

TabICLv2 uses one view with feature and class shuffling disabled. The
Mitra checkpoints run directly through the AutoGluon `Tab2D`
architecture. The classifier normalizes the logits for the two supplied
classes. For regression, the runner maps context targets to $[0,1]$ and
uses the 1,000-bin head with boundaries from $-0.5$ to $1.5$ following the
[Mitra implementation](https://huggingface.co/autogluon/mitra-finetune#distributional-regression).
It decodes the mean and quantiles under a uniform density within each
bin and converts them back to MPa. These are single-view, frozen-weight
runs on two particular splits.

### Data credits

The two CSVs are unmodified UCI snapshots and are licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The loader checks the source hashes and drops exact concrete duplicates
only in memory.

- [Raisin](https://doi.org/10.24432/C5660T) by İlkay Çinar, Murat Koklu,
  and Sakir Tasdemir (2020). [UCI source CSV](https://archive.ics.uci.edu/static/public/850/data.csv)
- [Concrete Compressive Strength](https://doi.org/10.24432/C5PK67) by
  I-Cheng Yeh (1998). [UCI source CSV](https://archive.ics.uci.edu/static/public/165/data.csv)

See the [11 exercises](EXERCISES.md) and
[complete solutions](../../solutions/ch29_tabular/README.md).
