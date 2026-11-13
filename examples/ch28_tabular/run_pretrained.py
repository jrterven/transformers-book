"""Evaluate frozen TabICLv2 and Mitra-v2 checkpoints on two UCI tables."""

import argparse
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from .model import brier_score, histogram_summary
from .tabular_data import DATA_SHA256, SEED, load_tables, split_table


TABICL_REPO = "jingang/TabICL"
TABICL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
CHECKPOINTS = {
    "tabicl_classifier": (
        TABICL_REPO, TABICL_REVISION, {
            "tabicl-classifier-v2-20260212.ckpt":
                "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
        },
    ),
    "tabicl_regressor": (
        TABICL_REPO, TABICL_REVISION, {
            "tabicl-regressor-v2-20260212.ckpt":
                "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
        },
    ),
    "mitra_classifier": (
        "autogluon/mitra-classifier-2", "edada0d20759c58ada8c8605c25f22f6e98ea5f0", {
            "config.json": "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df",
            "model.safetensors": "5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09",
        },
    ),
    "mitra_regressor": (
        "autogluon/mitra-regressor-2", "246c7d310fff04296a2388df5f448e3ffe809283", {
            "config.json": "f760c38c8884757f6141c1fd2e841e93a470be052dc5a4b489790097a002f0e6",
            "model.safetensors": "8c90b8f281a793f9feb19f8fc3c5529c58ac844dd7ffc690a0fdbb96fae0967c",
        },
    ),
}
QUANTILES = (0.1, 0.5, 0.9)
THRESHOLD = 0.5
MITRA_QUERY_BATCH = 48
TREE_OPTIONS = {
    "max_iter": 140, "max_leaf_nodes": 15, "min_samples_leaf": 5,
    "early_stopping": False, "random_state": SEED,
}


def checkpoint_paths(directory, key):
    """Resolve each checkpoint from the Hub or a supplied local mirror."""
    from huggingface_hub import hf_hub_download

    repo, revision, files = CHECKPOINTS[key]
    folder = "tabicl" if key.startswith("tabicl") else key.replace("_", "-")
    paths = []
    for name, expected_hash in files.items():
        if directory is None:
            path = Path(hf_hub_download(repo_id=repo, revision=revision, filename=name))
        else:
            path = directory / folder / name
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        if digest != expected_hash:
            raise ValueError(f"checkpoint hash mismatch: {path}")
        paths.append(path)
    return paths


def tabicl_prediction(checkpoint, x_train, y_train, x_test, classification, threads):
    from tabicl import TabICLClassifier, TabICLRegressor

    options = dict(
        model_path=checkpoint, allow_auto_download=False, device="cpu", n_estimators=1,
        norm_methods="none", feat_shuffle_method="none", n_jobs=threads, use_amp=False,
        use_fa3=False, random_state=SEED,
    )
    if classification:
        model = TabICLClassifier(class_shuffle_method="none", **options)
        model.fit(x_train, y_train)
        positive = np.flatnonzero(model.classes_ == 1).item()
        return model.predict_proba(x_test)[:, positive], None
    model = TabICLRegressor(**options)
    model.fit(x_train, y_train)
    result = model.predict(x_test, output_type=["mean", "quantiles"], alphas=list(QUANTILES))
    return result["mean"], result["quantiles"]


def mitra_prediction(model, x_train, y_train, x_test, classification):
    import torch

    support = torch.tensor(x_train, dtype=torch.float32)[None, ...]
    if classification:
        labels = torch.tensor(y_train, dtype=torch.long)
    else:
        minimum = float(y_train.min())
        span = float(y_train.max() - minimum)
        if span <= 0:
            raise ValueError("Mitra regression needs distinct support targets")
        edges = torch.linspace(-0.5, 1.5, model.dim_output + 1)
        normalized = torch.tensor((y_train - minimum) / span, dtype=torch.float32)
        labels = torch.bucketize(normalized, edges) - 1

    feature_mask = torch.zeros((1, support.shape[-1]), dtype=torch.bool)
    support_mask = torch.zeros((1, len(x_train)), dtype=torch.bool)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x_test), MITRA_QUERY_BATCH):
            query = torch.tensor(x_test[start:start + MITRA_QUERY_BATCH], dtype=torch.float32)
            query_mask = torch.zeros((1, len(query)), dtype=torch.bool)
            logits = model(
                support, labels[None, ...], query[None, ...],
                feature_mask, support_mask, query_mask,
            )[0]
            if classification:
                outputs.append(torch.softmax(logits[:, :2], dim=-1)[:, 1].numpy())
            else:
                outputs.append(torch.softmax(logits.double(), dim=-1).numpy())

    values = np.concatenate(outputs)
    if classification:
        return values, None
    means, quantiles = histogram_summary(edges.numpy(), values, QUANTILES)
    return minimum + span * means, minimum + span * quantiles


def baseline_predictions(x_train, y_train, x_test, classification):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    constant = np.full(len(x_test), y_train.mean())
    if classification:
        linear = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))
        trees = HistGradientBoostingClassifier(**TREE_OPTIONS)
        return {
            "constant": constant,
            "linear": linear.fit(x_train, y_train).predict_proba(x_test)[:, 1],
            "trees": trees.fit(x_train, y_train).predict_proba(x_test)[:, 1],
        }
    linear = make_pipeline(StandardScaler(), Ridge(alpha=1))
    trees = HistGradientBoostingRegressor(**TREE_OPTIONS)
    return {
        "constant": constant,
        "linear": linear.fit(x_train, y_train).predict(x_test),
        "trees": trees.fit(x_train, y_train).predict(x_test),
    }


def classification_metrics(actual, probability):
    from sklearn.metrics import roc_auc_score

    label = probability >= THRESHOLD
    return {
        "accuracy": float(np.mean(label == actual)),
        "auc": float(roc_auc_score(actual, probability)),
        "brier": brier_score(actual, probability),
        "tp": int(np.sum(label & (actual == 1))),
        "tn": int(np.sum(~label & (actual == 0))),
        "fp": int(np.sum(label & (actual == 0))),
        "fn": int(np.sum(~label & (actual == 1))),
    }


def regression_metrics(actual, prediction, quantiles):
    from sklearn.metrics import r2_score

    errors = actual - prediction
    scores = {
        "mae_mpa": float(np.mean(np.abs(errors))),
        "rmse_mpa": float(np.sqrt(np.mean(errors**2))),
        "r2": float(r2_score(actual, prediction)),
    }
    if quantiles is not None:
        low, high = quantiles[:, 0], quantiles[:, -1]
        covered = (low <= actual) & (actual <= high)
        scores.update(
            covered=int(covered.sum()), coverage=float(covered.mean()),
            width_mpa=float(np.mean(high - low)),
        )
    return scores


def score_prediction(actual, values, quantiles, classification):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != actual.shape or not np.isfinite(values).all():
        raise ValueError("model produced invalid point predictions")
    if quantiles is not None:
        quantiles = np.asarray(quantiles, dtype=np.float64)
        if (quantiles.shape != (len(actual), len(QUANTILES))
                or not np.isfinite(quantiles).all() or np.any(np.diff(quantiles, axis=1) < 0)):
            raise ValueError("model produced invalid or unordered quantiles")
    metrics = (classification_metrics(actual, values) if classification
               else regression_metrics(actual, values, quantiles))
    result = {"point": values.tolist(), "metrics": metrics}
    if quantiles is not None:
        result["quantiles"] = quantiles.tolist()
    return result


def run_table(name, table, checkpoint_dir, threads):
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    train, test = split_table(table)
    classification = name == "raisin"
    kind = "classifier" if classification else "regressor"
    tabicl, = checkpoint_paths(checkpoint_dir, "tabicl_" + kind)
    config, weights = checkpoint_paths(checkpoint_dir, "mitra_" + kind)
    if config.parent != weights.parent:
        raise ValueError("Mitra config and weights must share a checkpoint directory")
    mitra = Tab2D.from_pretrained(str(config.parent), device="cpu").eval()
    x_test = table.x[test]
    actual = table.y[test]
    result = {
        "source_rows": table.source_rows, "rows": len(table.y), "features": table.features,
        "training_ids": table.row_ids[train].tolist(), "test_ids": table.row_ids[test].tolist(),
        "actual": actual.tolist(), "runs": {},
    }
    if table.groups is not None:
        train_groups = np.unique(table.groups[train])
        test_groups = np.unique(table.groups[test])
        if np.intersect1d(train_groups, test_groups).size:
            raise ValueError("the split shares recipes across training and test rows")
        result.update(train_groups=len(train_groups), test_groups=len(test_groups))

    for n in (32, 128, len(train)):
        x_train, y_train = table.x[train[:n]], table.y[train[:n]]
        methods = baseline_predictions(x_train, y_train, x_test, classification)
        predictions = {
            method: score_prediction(actual, values, None, classification)
            for method, values in methods.items()
        }
        for method, predict in (
            ("tabicl", lambda: tabicl_prediction(tabicl, x_train, y_train, x_test, classification, threads)),
            ("mitra", lambda: mitra_prediction(mitra, x_train, y_train, x_test, classification)),
        ):
            print(f"{name}, {n} context rows, {method}", flush=True)
            point, quantiles = predict()
            predictions[method] = score_prediction(actual, point, quantiles, classification)
        result["runs"][str(n)] = predictions
    return result


def main():
    import torch
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, help="use a local mirror of the pinned checkpoint files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    tables = load_tables()
    with threadpool_limits(limits=args.threads):
        results = {
            name: run_table(name, table, args.checkpoints, args.threads)
            for name, table in tables.items()
        }
    report = {
        "seed": SEED, "threshold": THRESHOLD, "quantiles": QUANTILES, "device": "cpu",
        "threads": args.threads, "tabicl_views": 1, "mitra_query_batch": MITRA_QUERY_BATCH,
        "tree_options": TREE_OPTIONS, "source_sha256": DATA_SHA256, "checkpoints": CHECKPOINTS,
        "versions": {
            name: version(name) for name in (
                "autogluon.tabular", "numpy", "pandas", "scikit-learn", "scipy", "tabicl", "torch"
            )
        },
        "datasets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
