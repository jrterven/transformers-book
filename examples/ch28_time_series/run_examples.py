"""Run the Chapter 28 rolling forecasts and interval calculations."""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import (
    fit_seasonal_drift,
    forecast_metrics,
    rolling_windows,
    seasonal_drift,
    seasonal_naive,
)


def series_example():
    t = np.arange(32)
    return t + np.array([0, 3, -1, 2])[t % 4]


def rolling_example():
    values = series_example()
    split, lookback, horizon, period = 24, 4, 3, 4
    origins, _, targets = rolling_windows(values, lookback, horizon)
    train = origins + horizon <= split
    test = origins >= split
    learned_drift = fit_seasonal_drift(values[:split], period)
    naive, learned = [], []
    for origin in origins[test]:
        history = values[:origin]
        naive.append(seasonal_naive(history, period, horizon))
        learned.append(seasonal_drift(history, period, horizon, learned_drift))
    actual = targets[test]
    naive, learned = np.stack(naive), np.stack(learned)
    return {
        "split_index": split,
        "lookback": lookback,
        "horizon": horizon,
        "period": period,
        "training_origins": origins[train].tolist(),
        "omitted_crossing_origins": origins[~(train | test)].tolist(),
        "test_origins": origins[test].tolist(),
        "fitted_drift": learned_drift,
        "targets": actual.tolist(),
        "seasonal_naive": naive.tolist(),
        "seasonal_drift": learned.tolist(),
        "mae_seasonal_naive": np.abs(actual - naive).mean(axis=0).tolist(),
        "mae_seasonal_drift": np.abs(actual - learned).mean(axis=0).tolist(),
    }


def interval_example():
    actual = [[10, 12, 15], [11, 13, 14]]
    predicted = np.array([[10, 11, 14], [10, 13, 12]])
    lower, upper = predicted - 1, predicted + 1
    mae, coverage, width = forecast_metrics(actual, predicted, lower, upper)
    return {
        "actual": actual,
        "predicted": predicted.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "mae": mae.tolist(),
        "coverage": coverage.tolist(),
        "mean_interval_width": width.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save the JSON output to this path")
    args = parser.parse_args()
    result = {"rolling": rolling_example(), "intervals": interval_example()}
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
