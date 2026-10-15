import logging

import mlflow

logger = logging.getLogger(__name__)


def passes_quality_gate(
    metrics,
    config,
):

    minimum_accuracy = config[
        "evaluation"
    ]["minimum_accuracy"]

    accuracy = metrics["accuracy"]

    passed = accuracy >= minimum_accuracy
    logger.log(
        logging.INFO if passed else logging.WARNING,
        "QUALITY_GATE | %s | accuracy=%.4f | minimum_accuracy=%.4f",
        "Passed" if passed else "Failed", accuracy, minimum_accuracy,
    )

    mlflow.log_metric(
        "quality_gate_passed",
        int(passed),
    )

    mlflow.set_tag(
        "quality_gate",
        "passed" if passed else "failed",
    )

    logger.debug("QUALITY_GATE | Result logged to MLflow")
    return passed


def register_model(
    run_id,
    config,
    model_uri=None,
):

    model_name = config["mlflow"][
        "registered_model_name"
    ]

    model_uri = model_uri or f"runs:/{run_id}/model"

    logger.info(
        "REGISTRATION | Started | mlflow_run_id=%s | model_name=%s | model_uri=%s",
        run_id, model_name, model_uri,
    )
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    mlflow.set_tag(
        "registration_status",
        "registered",
    )

    logger.info(
        "REGISTRATION | Completed | model_name=%s | model_version=%s",
        registered_model.name, registered_model.version,
    )
    return registered_model
