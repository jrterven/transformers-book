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
