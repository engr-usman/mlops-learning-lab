"""Evaluate a registered model version on the held-out test set and promote it if it qualifies.

Usage:
    python evaluate.py [--version N] [--data path/to/test.csv] [--no-promote]

Metrics, the confusion matrix and the classification report are logged to a new
MLflow run. If the version meets mlflow.promotion_thresholds in config.yaml and its
ROC AUC is at least the current champion's, it gets the champion alias that api.py serves.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config_loader import load_config, setup_mlflow
from logger import get_logger

config = load_config()
logger = get_logger("evaluate")

TARGET = config["target"]["name"]
TARGET_CLASSES = config["target"]["classes"]
MODEL_NAME = config["mlflow"]["registered_model_name"]
CHAMPION_ALIAS = config["mlflow"]["champion_alias"]
DECISION_THRESHOLD = 0.5


def compute_metrics(y_true, y_proba) -> dict[str, float]:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def latest_version(client: MlflowClient) -> str:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise SystemExit(f"No versions of '{MODEL_NAME}' found; run train.py first")
    return str(max(int(v.version) for v in versions))


def champion_roc_auc(client: MlflowClient) -> float | None:
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None
    value = champion.tags.get("test_roc_auc")
    return float(value) if value is not None else None


def evaluate(version: str | None, data_path: Path, promote: bool) -> dict[str, float]:
    setup_mlflow()
    client = MlflowClient()

    version = version or latest_version(client)
    model_uri = f"models:/{MODEL_NAME}/{version}"
    logger.info("Loading %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == TARGET_CLASSES[1]).astype(int)
    logger.info("Evaluating on %d rows from %s", len(df), data_path)

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    metrics = compute_metrics(y, y_proba)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=TARGET_CLASSES, output_dict=True)

    with mlflow.start_run(run_name=f"evaluate-v{version}") as run:
        mlflow.set_tags({"stage": "evaluate", "model_version": version})
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("decision_threshold", DECISION_THRESHOLD)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"labels": TARGET_CLASSES, "matrix": cm.tolist()}, "confusion_matrix.json")
        mlflow.log_dict(report, "classification_report.json")
        logger.info("MLflow run %s", run.info.run_id)

    for name, value in metrics.items():
        logger.info("%-9s %.4f", name, value)
    logger.info("Confusion matrix (rows=actual, cols=predicted, %s): %s", TARGET_CLASSES, cm.tolist())

    for name, value in metrics.items():
        client.set_model_version_tag(MODEL_NAME, version, f"test_{name}", f"{value:.4f}")

    if promote:
        maybe_promote(client, version, metrics)
    return metrics


def maybe_promote(client: MlflowClient, version: str, metrics: dict[str, float]) -> None:
    thresholds = config["mlflow"]["promotion_thresholds"]
    failed = {k: (metrics[k], t) for k, t in thresholds.items() if metrics[k] < t}
    if failed:
        for k, (value, threshold) in failed.items():
            logger.warning("Not promoting: %s=%.4f is below threshold %.4f", k, value, threshold)
        return

    current = champion_roc_auc(client)
    if current is not None and metrics["roc_auc"] < current:
        logger.warning("Not promoting: roc_auc=%.4f is below current champion's %.4f",
                       metrics["roc_auc"], current)
        return

    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    logger.info("Promoted version %s to '%s'", version, CHAMPION_ALIAS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="model version to evaluate (default: latest)")
    parser.add_argument("--data", type=Path, default=config["paths"]["test_data_path"])
    parser.add_argument("--no-promote", dest="promote", action="store_false",
                        help="only evaluate; never move the champion alias")
    args = parser.parse_args()
    evaluate(args.version, args.data, args.promote)
