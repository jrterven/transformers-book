"""Replay fixed rolling origins with TimesFM 3, Timestral and seasonal baselines."""

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import numpy as np

from .demand_data import AREAS, HOUR, SNAPSHOT, SOURCE_URL, STAMP, load_snapshot
from .model import forecast_metrics, seasonal_naive


LOOKBACK = 28 * 24
HORIZON = 24
REPORT_LAG = 1
LEVELS = (0.1, 0.5, 0.9)
PERIODS = {"daily": 24, "weekly": 168}
CHECKPOINTS = {
    "timesfm3": {
        "repo": "google/timesfm-3.0-pytorch",
        "revision": "43046b85ec22d584a13f8098c2ed39c889e129c2",
        "code_revision": "918df02c9d366b2145bc5876135fa2131d3ed2a6",
        "weights_sha256": "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da",
        "config_sha256": "ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6",
    },
    "timestral": {
        "repo": "mistralai/Timestral-28M",
        "revision": "49b26409ac87b5935539a5d8ffe7f397b720db47",
        "code_revision": "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff",
        "weights_sha256": "dfc83c6220937ce5fc6687b35a4df94cd25e28a2cfe6fb92a23dad6301b2015d",
        "config_sha256": "0cdf44b8ea981191de53078e0a5ecbbd437046bd939a546d5dbe17868f71281f",
    },
}


def file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def checkpoint_path(key, local_root, offline):
    from huggingface_hub import snapshot_download

    entry = CHECKPOINTS[key]
    if local_root is not None:
        path = local_root / key
    else:
        path = Path(snapshot_download(entry["repo"], revision=entry["revision"],
                                      allow_patterns=["config.json", "model.safetensors"],
                                      local_files_only=offline))
    for filename, field in (("config.json", "config_sha256"), ("model.safetensors", "weights_sha256")):
        if file_hash(path / filename) != entry[field]:
            raise ValueError(f"Unexpected checkpoint file {key}/{filename}")
    return path


def issue_hours(first_day, count):
    start = datetime.fromisoformat(first_day).replace(tzinfo=timezone.utc)
    return tuple(start + timedelta(days=i) for i in range(count))


# snippet:demand_windows
def demand_windows(values, issue_indices):
    """Build observed histories and scored targets around midnight UTC issues."""
    issue = np.asarray(issue_indices)[:, None]
    history_index = issue + np.arange(1 - REPORT_LAG - LOOKBACK, 1 - REPORT_LAG)
    target_index = issue + np.arange(1, HORIZON + 1)
    return values[history_index], values[target_index]
# end-snippet:demand_windows


def calendar_rows(hours, timezone_name):
    """Describe the local start of each hour using only its timestamp."""
    tz = ZoneInfo(timezone_name)
    local_start = [(stamp - HOUR).astimezone(tz) for stamp in hours]
    angle = (2 * np.pi / 24) * np.array([stamp.hour for stamp in local_start])
    weekend = [stamp.weekday() >= 5 for stamp in local_start]
    return np.array([np.sin(angle), np.cos(angle), weekend], dtype=np.float32)


def build_replay(hours, values, timezone_name):
    calibration = issue_hours("2026-10-16", 13)
    test = issue_hours("2026-10-30", 14)
    if calibration[-1] + (HORIZON + REPORT_LAG) * HOUR > test[0]:
        raise ValueError("The last calibration target must have arrived before testing")
    position = {stamp: i for i, stamp in enumerate(hours)}
    indices = np.array([position[stamp] for stamp in calibration + test])
    if indices[0] < LOOKBACK + REPORT_LAG - 1 or indices[-1] + HORIZON >= len(values):
        raise ValueError("Every issue needs a complete history and forecast window")
    histories, targets = demand_windows(values, indices)
    calendar = calendar_rows(hours, timezone_name)
    covariates = [calendar[:, i + 1 - REPORT_LAG - LOOKBACK:i + HORIZON + 1] for i in indices]
    split = len(calibration)
    return {"calibration_issues": calibration, "test_issues": test,
            "calibration_history": histories[:split], "calibration_actual": targets[:split],
            "history": histories[split:], "actual": targets[split:], "calendar": covariates[split:]}


def summarize(actual, median, lower=None, upper=None):
    if lower is None and upper is None:
        mae = np.abs(actual - median).mean(axis=0)
        return {"mae_by_hour_mw": mae.tolist(), "mae_mw": float(mae.mean())}
    mae, coverage, width = forecast_metrics(actual, median, lower, upper)
    alpha = 2 * LEVELS[0]
    score = (upper - lower) + (2 / alpha) * (np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))
    return {"mae_by_hour_mw": mae.tolist(), "coverage_by_hour": coverage.tolist(),
            "width_by_hour_mw": width.tolist(), "interval_score_by_hour_mw": score.mean(axis=0).tolist(),
            "mae_mw": float(mae.mean()), "coverage": float(coverage.mean()),
            "width_mw": float(width.mean()), "interval_score_mw": float(score.mean()),
            "below": int((actual < lower).sum()), "above": int((actual > upper).sum())}


def baseline_rows(history, period):
    return np.stack([seasonal_naive(row, period, REPORT_LAG + HORIZON)[REPORT_LAG:] for row in history])


def baselines(replay):
    actual = replay["actual"]
    daily = baseline_rows(replay["history"], PERIODS["daily"])
    weekly = baseline_rows(replay["history"], PERIODS["weekly"])
    calibration_prediction = baseline_rows(replay["calibration_history"], PERIODS["weekly"])
    residual = replay["calibration_actual"] - calibration_prediction
    offsets = np.quantile(residual, [LEVELS[0], LEVELS[-1]], axis=0, method="linear")
    lower, upper = weekly + offsets[0], weekly + offsets[1]
    return {
        "daily": {"median_mw": daily.tolist(), "metrics": summarize(actual, daily)},
        "weekly": {"median_mw": weekly.tolist(), "lower_mw": lower.tolist(), "upper_mw": upper.tolist(),
                   "residual_offsets_mw": offsets.tolist(), "metrics": summarize(actual, weekly, lower, upper)},
    }


def predict_timesfm(forecaster, histories, covariates):
    forecasts = forecaster.predict_batch(
        list(histories), horizon=REPORT_LAG + HORIZON, past_future_covariates=covariates,
        return_quantiles=True, sort_quantiles=False,
    )
    return np.stack([forecast.quantiles for forecast in forecasts]), forecaster.config.quantiles


def predict_timestral(forecaster, histories, covariates):
    # Match the number of independent series per CPU batch to TimesFM's wrapper.
    forecasts = []
    for start in range(0, len(histories), 4):
        batch = slice(start, start + 4)
        calendar = None if covariates is None else np.stack(covariates[batch])
        forecast = forecaster.forecast(histories[batch], prediction_length=REPORT_LAG + HORIZON,
                                       known_covariates=calendar, symmetric_averaging=False,
                                       clip_nonnegative=False)
        forecasts.append(forecast.quantiles[:, 0].numpy())
    return np.concatenate(forecasts), forecaster.quantile_levels


def score_quantiles(raw, levels, actual):
    raw = raw[:, REPORT_LAG:]
    if raw.shape != (*actual.shape, len(levels)) or not np.isfinite(raw).all():
        raise ValueError("Model output must be finite and align with scored origins and hours")
    crossings = int(np.any(np.diff(raw, axis=-1) < 0, axis=-1).sum())
    ordered = np.sort(raw, axis=-1)
    lower, median, upper = (ordered[:, :, levels.index(q)] for q in LEVELS)
    return {"raw_crossing_slots": crossings, "native_levels": list(levels),
            "lower_mw": lower.tolist(), "median_mw": median.tolist(), "upper_mw": upper.tolist(),
            "metrics": summarize(actual, median, lower, upper)}


def main(args):
    import torch
    from timesfm3.torch import TimesFM3Forecaster
    from timestral import TimestralForecaster

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    hours, demand = load_snapshot(args.data)
    replays = {area: build_replay(hours, values, AREAS[area]) for area, values in demand.items()}
    report = {"source": SOURCE_URL, "snapshot_sha256": file_hash(args.data),
              "lookback_hours": LOOKBACK, "report_lag_hours": REPORT_LAG, "horizon_hours": HORIZON,
              "quantile_levels": LEVELS, "models": CHECKPOINTS, "areas": {}, "device": "cpu", "dtype": "float32",
              "periods_hours": PERIODS, "forecast_batch_size": 4, "symmetric_averaging": False,
              "calendar": ["sin_local_hour", "cos_local_hour", "local_weekend"],
              "platform": platform.platform(),
              "versions": {key: version(key) for key in ("numpy", "torch", "timesfm", "timestral")}}
    for area, replay in replays.items():
        report["areas"][area] = {"timezone": AREAS[area],
                                 "test_issues_utc": [t.strftime(STAMP) for t in replay["test_issues"]],
                                 "calibration_issues_utc": [t.strftime(STAMP) for t in replay["calibration_issues"]],
                                 "actual_mw": replay["actual"].tolist(), "forecasts": baselines(replay)}

    for key, cls, predict in (("timesfm3", TimesFM3Forecaster, predict_timesfm),
                              ("timestral", TimestralForecaster, predict_timestral)):
        path = checkpoint_path(key, args.model_root, args.offline)
        forecaster = cls.from_pretrained(str(path), device="cpu", local_files_only=True)
        for area, replay in replays.items():
            history = replay["history"].astype(np.float32)
            for suffix, covariates in (("", None), ("_calendar", replay["calendar"])):
                print(f"{key}{suffix} · {area}", flush=True)
                raw, levels = predict(forecaster, history, covariates)
                report["areas"][area]["forecasts"][key + suffix] = score_quantiles(raw, levels, replay["actual"])
        del forecaster

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SNAPSHOT)
    parser.add_argument("--model-root", type=Path, help="optional local directories timesfm3/ and timestral/")
    parser.add_argument("--offline", action="store_true", help="only use already downloaded weights")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
