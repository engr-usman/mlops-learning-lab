"""Four file-based stages; no active run or DataFrame survives between processes."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd

from src.artifact_io import check_hash, read_json, sha256, write_json
from src.config import load_config
from src.data_validation import validate_dataset
from src.evaluation import evaluate_model, select_best_model
from src.registration import passes_quality_gate
from src.training import create_pipeline, split_dataset, train_model

LOGGER = logging.getLogger(__name__)
ROW_ID = "__source_row_id"
MODELS = ("logistic_regression", "decision_tree", "random_forest", "gradient_boosting")


def check_config(config: dict[str, Any]) -> None:
    """Fail early on unsupported contracts while preserving bank business rules."""
    data = config["data"]
    features = data["features"]
    if not features or len(features) != len(set(features)):
        raise ValueError("Features must be nonempty and unique")
    if data["target"] in features or ROW_ID in features or data["target"] == ROW_ID:
        raise ValueError("Target/reserved row ID must not be used as a feature")
    if set(data["target_values"]) != {"yes", "no"} or data["positive_label"] != "yes":
        raise ValueError("This bank pipeline requires binary no/yes labels; yes is positive")
    if set(config["models"]) != set(MODELS):
        raise ValueError("Exactly the four original classifiers must be configured")
    if not 0 < config["training"]["test_size"] < 1:
        raise ValueError("test_size must be strictly between zero and one")
    if not 0 <= config["evaluation"]["minimum_accuracy"] <= 1:
        raise ValueError("minimum_accuracy must be between zero and one")
    if config["evaluation"]["selection_metric"] != "f1_score":
        raise ValueError("This bank pipeline selects by f1_score")
    # Reuse the original feature-partition checks without fitting anything.
    create_pipeline(config, next(iter(config["models"])))


def validate_stage(dataset_path: str, config_path: str, validated_path: str,
                   snapshot_path: str, report_path: str) -> dict[str, Any]:
    """Validate source CSV, materialize a Dataset and freeze configuration/lineage."""
    config = load_config(config_path)
    check_config(config)
    data = config["data"]
    frame = validate_dataset(dataset_path, data["features"], data["target"],
                             data["numeric_features"], data["target_values"],
                             config["training"]["test_size"])
    if ROW_ID in frame.columns:
        raise ValueError(f"Dataset must not contain reserved column {ROW_ID}")
    # Exercise the exact splitter without fitting models; catches actual splitter
    # failures as well as the original validator's count/size prechecks.
    split_dataset(frame, config)
    Path(validated_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(validated_path, index=False)
    write_json(snapshot_path, config)
    report = {
        "schema_version": 1, "validation_status": "PASSED", "rows": len(frame),
        "columns": len(frame.columns), "feature_count": len(data["features"]),
        "source_dataset_sha256": sha256(dataset_path),
        "validated_dataset_sha256": sha256(validated_path),
        "config_sha256": sha256(snapshot_path),
        "class_counts": {str(k): int(v) for k, v in frame[data["target"]].value_counts().items()},
        "features": data["features"], "target": data["target"],
    }
    write_json(report_path, report)
    LOGGER.info("VALIDATE | rows=%d | hash=%s", len(frame), report["source_dataset_sha256"])
    return report


def index_hash(indices: list[int]) -> str:
    return hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()


def train_stage(validated_path: str, snapshot_path: str, validation_path: str,
                tracking_uri: str, execution_id: str, manifest_path: str,
                heldout_path: str, split_path: str) -> dict[str, Any]:
    """Fit four models on one split and persist actual MLflow model URIs."""
    report = read_json(validation_path)
    if report["validation_status"] != "PASSED":
        raise ValueError("Input dataset did not pass validation")
    check_hash(validated_path, report["validated_dataset_sha256"])
    check_hash(snapshot_path, report["config_sha256"])
    config = read_json(snapshot_path)
    check_config(config)
    config["data"]["path"] = validated_path  # train_model hashes the artifact, never the source CSV.
    frame = pd.read_csv(validated_path)
    split = split_dataset(frame, config)
    X_train, X_test, y_train, y_test = split
    train_indices = [int(i) for i in X_train.index]
    test_indices = [int(i) for i in X_test.index]
    if set(train_indices) & set(test_indices):
        raise ValueError("Training and held-out rows overlap")
    heldout = X_test.copy()
    heldout[config["data"]["target"]] = y_test
    heldout[ROW_ID] = test_indices
    Path(heldout_path).parent.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(heldout_path, index=False)
    split_metadata = {
        "schema_version": 1, "training_rows": len(X_train), "test_rows": len(X_test),
        "random_seed": config["project"]["random_state"],
        "test_size": config["training"]["test_size"], "stratified": True,
        "train_row_ids": train_indices, "test_row_ids": test_indices,
        "train_indices_sha256": index_hash(train_indices),
        "test_indices_sha256": index_hash(test_indices),
        "heldout_sha256": sha256(heldout_path), **report,
    }
    write_json(split_path, split_metadata)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(config["mlflow"]["experiment_name"])
    candidates = []
    with mlflow.start_run(run_name=f"four-stage-{execution_id}", tags={
        "four_stage_execution_id": execution_id, "workflow_state": "training",
        "architecture": "kfp-v2-four-stage", "source_dataset_sha256": report["source_dataset_sha256"],
    }) as parent:
        mlflow.log_dict(report, "validation.json")
        mlflow.log_dict(config, "configuration.json")
        mlflow.log_dict(split_metadata, "split_metadata.json")
        for name in config["models"]:
            with mlflow.start_run(run_name=name, nested=True, tags={
                "model_type": name, "four_stage_execution_id": execution_id,
                "source_dataset_sha256": report["source_dataset_sha256"],
                "heldout_sha256": split_metadata["heldout_sha256"],
                "workflow_state": "training",
            }) as child:
                result = train_model(frame, config, mlflow_run=child, model_name=name, split=split)
                uri = result["model_uri"]
                if not uri:
                    raise ValueError("MLflow did not return a model URI")
                candidates.append({"model_name": name, "run_id": child.info.run_id,
                                   "model_uri": uri, "heldout_sha256": split_metadata["heldout_sha256"]})
                mlflow.set_tag("workflow_state", "awaiting_evaluation")
            LOGGER.info("TRAIN | model=%s | run_id=%s | uri=%s", name, child.info.run_id, uri)
        manifest = {
            "schema_version": 1, "execution_id": execution_id,
            "parent_run_id": parent.info.run_id, "experiment_id": experiment.experiment_id,
            "candidates": candidates, "heldout_sha256": split_metadata["heldout_sha256"],
            "split_metadata_sha256": sha256(split_path), "config_sha256": report["config_sha256"],
            "source_dataset_sha256": report["source_dataset_sha256"],
        }
        mlflow.log_dict(manifest, "candidate_manifest.json")
        mlflow.set_tag("workflow_state", "awaiting_evaluation")
    write_json(manifest_path, manifest)
    return manifest


def evaluate_stage(manifest_path: str, heldout_path: str, split_path: str,
                   snapshot_path: str, tracking_uri: str, evaluation_path: str) -> dict[str, Any]:
    """Reload models from MLflow and resume their explicit runs in a fresh process."""
    manifest = read_json(manifest_path)
    check_hash(heldout_path, manifest["heldout_sha256"])
    check_hash(split_path, manifest["split_metadata_sha256"])
    check_hash(snapshot_path, manifest["config_sha256"])
    config = read_json(snapshot_path)
    split_metadata = read_json(split_path)
    check_config(config)
    frame = pd.read_csv(heldout_path)
    if frame[ROW_ID].tolist() != split_metadata["test_row_ids"]:
        raise ValueError("Held-out row IDs do not match the original split")
    candidates = manifest["candidates"]
    if [c["model_name"] for c in candidates] != list(config["models"]):
        raise ValueError("Candidate manifest order/models differ from configuration")
    X_test = frame[config["data"]["features"]]
    y_test = frame[config["data"]["target"]]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    evaluated = []
    with mlflow.start_run(run_id=manifest["parent_run_id"]):
        mlflow.set_tag("workflow_state", "evaluating")
        for candidate in candidates:
            if candidate["heldout_sha256"] != manifest["heldout_sha256"]:
                raise ValueError("Candidates must share the exact held-out artifact")
            run = client.get_run(candidate["run_id"])
            if run.data.tags.get("mlflow.parentRunId") != manifest["parent_run_id"]:
                raise ValueError("Candidate is not a child of this parent run")
            with mlflow.start_run(run_id=candidate["run_id"], nested=True):
                model = mlflow.sklearn.load_model(candidate["model_uri"])
                metrics = evaluate_model(model, X_test, y_test, config["data"]["positive_label"])
                eligible = passes_quality_gate(metrics, config)
                predictions = model.predict(X_test)
                from sklearn.metrics import confusion_matrix
                matrix = confusion_matrix(y_test, predictions, labels=["no", "yes"]).tolist()
                evaluated.append({**candidate, "metrics": metrics, "eligible": eligible,
                                  "confusion_matrix": {"labels": ["no", "yes"], "counts": matrix}})
                mlflow.set_tag("workflow_state", "evaluated")
            LOGGER.info("EVALUATE | model=%s | metrics=%s | eligible=%s",
                        candidate["model_name"], metrics, eligible)
        winner = select_best_model(evaluated, config)
        result = {
            **{key: manifest[key] for key in ("parent_run_id", "experiment_id", "execution_id",
                                             "config_sha256", "source_dataset_sha256")},
            "schema_version": 1, "candidates": evaluated, "selected_candidate": winner,
            "quality_gate": {"passed": winner is not None,
                             "minimum_accuracy": config["evaluation"]["minimum_accuracy"],
                             "selection_metric": config["evaluation"]["selection_metric"]},
            "heldout_sha256": manifest["heldout_sha256"],
        }
        mlflow.log_metrics({f"{c['model_name']}_{k}": v
                            for c in evaluated for k, v in c["metrics"].items()})
        mlflow.log_dict(result, "model_comparison.json")
        mlflow.log_params(config["evaluation"])
        mlflow.set_tag("quality_gate", "passed" if winner else "failed")
        mlflow.set_tag("workflow_state", "awaiting_registration")
        if winner:
            mlflow.log_metrics(winner["metrics"])
            mlflow.set_tags({"selected_model": winner["model_name"], "selected_run_id": winner["run_id"]})
    write_json(evaluation_path, result)
    return result


def find_registered_version(client: MlflowClient, name: str, key: str,
                            run_id: str, model_uri: str) -> Any | None:
    """Search all pages for an atomically tagged registration of this winner."""
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    page_token = None
    while True:
        page = client.search_model_versions(f"name = '{escaped}'", max_results=100,
                                            page_token=page_token)
        for version in page:
            if (version.tags.get("four_stage_registration_key") == key
                    and version.run_id == run_id
                    and version.tags.get("selected_model_uri") == model_uri):
                return version
        page_token = getattr(page, "token", None)
        if not page_token:
            return None


def register_stage(evaluation_path: str, snapshot_path: str, tracking_uri: str,
                   registration_path: str) -> dict[str, Any]:
    """Register an eligible winner; sequential retries reuse a tagged version.

    This is best-effort idempotency, not a distributed lock. Concurrent requests
    can race between search/create; this pipeline does not automatically retry
    training or run parallel registration tasks for the same evaluation artifact.
    """
    evaluation = read_json(evaluation_path)
    check_hash(snapshot_path, evaluation["config_sha256"])
    config = read_json(snapshot_path)
    check_config(config)
    candidates = evaluation["candidates"]
    recomputed = select_best_model(candidates, config)
    winner = evaluation["selected_candidate"]
    passed = evaluation["quality_gate"]["passed"]
    if winner != recomputed or passed != (recomputed is not None):
        raise ValueError("Selection and quality gate are inconsistent with candidate metrics")
    if evaluation["quality_gate"]["minimum_accuracy"] != config["evaluation"]["minimum_accuracy"]:
        raise ValueError("Quality threshold changed after evaluation")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    name = config["mlflow"]["registered_model_name"]
    result = {
        "schema_version": 1, "status": "not_registered", "registered_model_name": name,
        "registered_model_version": None, "parent_run_id": evaluation["parent_run_id"],
        "experiment_id": evaluation["experiment_id"], "execution_id": evaluation["execution_id"],
        "selected_model": None, "selected_run_id": None, "selected_model_uri": None,
        "summary_metrics": {}, "quality_gate": evaluation["quality_gate"],
    }
    with mlflow.start_run(run_id=evaluation["parent_run_id"]):
        if winner:
            child = client.get_run(winner["run_id"])
            if child.data.tags.get("mlflow.parentRunId") != evaluation["parent_run_id"]:
                raise ValueError("Selected source run has incorrect parent lineage")
            key = hashlib.sha256(f"{name}\0{evaluation['parent_run_id']}\0{winner['model_uri']}".encode()).hexdigest()
            version = find_registered_version(client, name, key, winner["run_id"], winner["model_uri"])
            reused = version is not None
            if version is None:
                # tags are submitted with creation, closing the post-create/tag retry gap.
                version = mlflow.register_model(winner["model_uri"], name, tags={
                    "four_stage_registration_key": key, "selected_model_uri": winner["model_uri"],
                    "four_stage_parent_run_id": evaluation["parent_run_id"],
                    "selected_model": winner["model_name"],
                })
            # Verify the registry object and actual fitted model before reporting completion.
            version = client.get_model_version(name, str(version.version))
            if version.run_id != winner["run_id"]:
                raise ValueError("Registered model has incorrect source run")
            mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
            result.update({"status": "registered", "registered_model_version": str(version.version),
                           "selected_model": winner["model_name"], "selected_run_id": winner["run_id"],
                           "selected_model_uri": winner["model_uri"], "summary_metrics": winner["metrics"],
                           "registration_key": key, "reused_existing_version": reused})
        else:
            result["reason"] = "No candidate passed the minimum accuracy gate"
        mlflow.set_tags({"workflow_state": "completed", "registration_status": result["status"]})
        if result["registered_model_version"]:
            mlflow.set_tag("registered_model_version", result["registered_model_version"])
        mlflow.log_dict(result, "registration_result.json")
    write_json(registration_path, result)
    LOGGER.info("REGISTER | result=%s", result)
    return result
