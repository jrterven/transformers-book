# Chapter 27. Time-Series Forecasting

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch27_time_series.run_examples --output output/ch27.json
python -m unittest discover -s tests -p test_time_series.py -v
```

## Build windows in chronological order

`rolling_windows` uses a one-dimensional series and returns origins,
past rows, and future target rows. Origin `o` is the first unknown
array index. A row reads the preceding `lookback` values and scores
the next `horizon` values. Both parts must fit inside the series. An
input with no complete windows yields empty row axes.

The fixture uses a split at index 24. Training targets end before
that boundary, and test targets start at or after it. The example
omits windows whose targets cross the split. At each test origin,
the forecast can use values observed before that origin.

## Compare two seasonal baselines

`seasonal_naive` repeats the latest observed period. The learned
baseline uses `fit_seasonal_drift` to average lagged changes on the
training prefix. `seasonal_drift` adds that fixed estimate once in
the first forecast cycle, twice in the second, and so on. The
rolling example expands the observed history with each origin and
keeps the fitted drift fixed.

## Report one metric per horizon

`forecast_metrics` takes actual values, point predictions, and
lower and upper interval endpoints. The four arrays share shape
`[origins, horizon]`. The function averages absolute errors,
interval hit indicators, and widths over the origins. Coverage
includes values on either endpoint. The interval fixture shows
how the third horizon can have a larger error and lower coverage
even when widths are the same across all horizons.

## Keep time and uncertainty distinct

There are six test origins, 18 forecast-target pairs and eight distinct target
timestamps in the rolling fixture. A value may be scored at several horizons.
The fitted drift stays fixed while the input history is refreshed.

Quantile loss penalizes the two error directions differently. Coverage counts
outcomes within issued intervals; width describes their precision. Increasing
the width leaves point MAE unchanged. Choose interval rules before test outcomes
arrive and preserve the versions of covariates available at each origin.

See the [11 exercises](EXERCISES.md) and [complete solutions](../../solutions/ch27_time_series/README.md).

## Replay 24-hour electricity demand forecasts

Source: U.S. Energy Information Administration, [Hourly Electric Grid Monitor](https://www.eia.gov/electricity/930-content/about.html), EIA-930 balance file for July–December 2026, snapshot downloaded November 13, 2026. The [checked-in extract](data/eia930_demand_2026.csv) contains reported New York (`NYIS`) and Texas (`ERCO`) demand in MW from September 18 at 00:00 through November 13 at 00:00 UTC. It has 1,345 consecutive hour-ending timestamps. [The extraction script](demand_data.py) checks duplicates, completeness, and imputation flags. The original balance-file SHA-256 at download was `945cd62a3edeb99c56de8c57d708772a56a8bc85b6b188d49d8db02ae74076ad`. The archive can receive revisions.

At 00:00 UTC each method reads 672 hours of history through the hour ending
at 23:00. The replay assumes the one-hour publication lag described by EIA.
It requests 25 steps and scores the last 24, skipping the 00:00 hour that
falls between the history and the scored day. The 14 test issues run from
October 30 through November 12. The weekly baseline builds interval offsets
from the 0.1 and 0.9 quantiles of signed errors at each horizon over 13
earlier issues from October 16 through 28, with linear interpolation.
The daily baseline supplies point predictions. Seasonal periods are 24 and
168 UTC hours, including over the November 1 clock change.

The pretrained forecasts use fixed weights in float32 on CPU. Both models
run once with demand history alone and once with local hour-of-day sine and
cosine and a weekend indicator. These covariates describe the start of each
hour and are available throughout the history and forecast windows. The
implementation checks for quantile crossing and orders the quantiles before
scoring. None of the 2,688 scored model slots cross at native quantile levels.

| Method | New York MAE (MW) | Texas MAE (MW) |
| --- | ---: | ---: |
| Daily repeat | 771 | 2,050 |
| Weekly repeat | 590 | 3,226 |
| TimesFM 3 | 427 | 1,540 |
| TimesFM 3 + calendar | 410 | 1,473 |
| Timestral-28M | 432 | 1,440 |
| Timestral-28M + calendar | 414 | 1,479 |

The scores describe these 14 issues in each grid. See the [forecast record](records/forecast_runs.json)
for the point and interval predictions, actuals, per-horizon metrics,
coverage, width, interval score, and the exact checkpoint and software revisions.

## Rebuild figures and rerun inference

From the repository root, rebuild the figures without model downloads.

```sh
python -m pip install numpy==2.5.0 matplotlib==3.11.2
python -m examples.ch27_time_series.make_figures \
  --record examples/ch27_time_series/records/forecast_runs.json \
  --output output/ch27/figures
```

For a new run, use Python 3.12 and install the pinned inference dependencies.
The first run downloads about 1.4 GB of weights and checks the hashes of the
weights and configs. `--offline` uses cached checkpoints, and `--model-root`
accepts local directories named `timesfm3` and `timestral`.

```sh
python -m pip install -r requirements-ch27-models.txt
python -m examples.ch27_time_series.run_pretrained --output output/ch27/forecast_runs.json
python scripts/export_snippets.py --chapter ch27 --check
```
