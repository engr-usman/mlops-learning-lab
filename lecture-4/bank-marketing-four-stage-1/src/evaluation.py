import logging

import mlflow
from matplotlib.figure import Figure

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


logger = logging.getLogger(__name__)


def evaluate_model(
    pipeline,
    X_test,
    y_test,
    positive_label="yes",
):

    logger.info("EVALUATION | Started | test_rows=%d", len(X_test))
    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label=positive_label,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=positive_label,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=positive_label,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    logger.info(
        "EVALUATION | Metrics calculated | accuracy=%.4f | precision=%.4f | recall=%.4f | f1_score=%.4f",
        accuracy, precision, recall, f1,
    )
    logger.debug("EVALUATION | Logging metrics and stage tag to MLflow")
    mlflow.log_metrics(metrics)

    labels = ["no", "yes"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    figure = Figure(figsize=(6, 5), layout="constrained")
    axis = figure.subplots()
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=axis, cmap="Blues", colorbar=False
    )
    axis.set_title("Held-out confusion matrix")
    try:
        mlflow.log_figure(figure, "confusion_matrix.png")
    finally:
        figure.clear()
    mlflow.log_dict({"labels": labels, "rows": "actual", "columns": "predicted",
                     "counts": matrix.tolist()}, "confusion_matrix.json")

    mlflow.set_tag(
        "pipeline_stage",
        "evaluation",
    )

    logger.info("EVALUATION | Completed | metrics logged to MLflow")
    return metrics


def select_best_model(candidates, config):
    """Rank eligible candidates by F1, then accuracy; preserve order for exact ties."""
    metric = config["evaluation"]["selection_metric"]
    eligible = [candidate for candidate in candidates
                if candidate["metrics"]["accuracy"] >= config["evaluation"]["minimum_accuracy"]]
    return max(eligible, key=lambda candidate: (
        candidate["metrics"][metric], candidate["metrics"]["accuracy"]
    ), default=None)
