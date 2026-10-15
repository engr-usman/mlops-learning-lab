import logging
import uuid
from contextlib import contextmanager
from time import perf_counter

import mlflow

from src.logging_config import run_logging_context, setup_logging
from src.config import load_config
from src.data_validation import validate_dataset
from src.training import train_model, split_dataset
from src.evaluation import evaluate_model, select_best_model
from src.registration import (
    passes_quality_gate,
    register_model,
)


logger = logging.getLogger(__name__)


@contextmanager
def pipeline_stage(summary, stage):
    """Stop on stage errors, preserving the original exception if tracking fails."""
    summary[stage] = "RUNNING"
    try:
        yield
    except Exception as error:
        summary[stage] = "FAILED"
        logger.exception("PIPELINE | Stage failed | stage=%s", stage)
        try:
            mlflow.set_tags({
                "failed_stage": stage.lower().replace(" ", "_"),
                "failure_type": type(error).__name__,
            })
        except Exception:
            logger.warning(
                "PIPELINE | Could not record failure tags | stage=%s",
                stage,
                exc_info=True,
            )
        raise
    else:
        if summary[stage] == "RUNNING":
            summary[stage] = "PASSED"


def log_execution_summary(summary):
    """Emit a single summary record using the project's shared log format."""
    separator = "=" * 50
    lines = [separator, "PIPELINE EXECUTION SUMMARY", separator]
    lines.extend(f"{label:<16}: {value}" for label, value in summary.items())
    lines.append(separator)
    logger.info("\n%s", "\n".join(lines))


def run_pipeline():
    setup_logging()
    pipeline_run_id = str(uuid.uuid4())
    with run_logging_context(pipeline_run_id):
        summary = {
            "Pipeline Run ID": pipeline_run_id,
            "MLflow Run ID": "N/A",
            "Data Validation": "SKIPPED",
            "Training": "SKIPPED",
            "Evaluation": "SKIPPED",
            "Accuracy": "N/A",
            "Quality Gate": "SKIPPED",
            "Registration": "SKIPPED",
        }
        started = perf_counter()
        logger.info("PIPELINE | Started")
        try:
            result = _run_pipeline(pipeline_run_id, summary)
        except Exception:
            log_failure = (
                logger.error if "FAILED" in summary.values() else logger.exception
            )
            log_failure(
                "PIPELINE | Failed | duration_seconds=%.3f",
                perf_counter() - started,
            )
            raise
        else:
            logger.info(
                "PIPELINE | Completed | mlflow_run_id=%s | status=%s | duration_seconds=%.3f",
                result["run_id"], result["status"], perf_counter() - started,
            )
            logger.debug("PIPELINE | Result | result=%s", result)
            return result
        finally:
            log_execution_summary(summary)


def _run_pipeline(pipeline_run_id, summary):

    config = load_config()
    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    logger.info(
        "PIPELINE | MLflow configured | experiment_name=%s",
        config["mlflow"]["experiment_name"],
    )
    with mlflow.start_run(
        run_name=f"pipeline-run-{pipeline_run_id}",
        tags={"pipeline_run_id": pipeline_run_id},
    ) as run:
        summary["MLflow Run ID"] = run.info.run_id
        logger.info(
            "PIPELINE | MLflow run linked | mlflow_run_id=%s",
            run.info.run_id,
        )
        logger.info("PIPELINE | Stage 1: Validating dataset")

        with pipeline_stage(summary, "Data Validation"):
            data_config = config["data"]
            df = validate_dataset(
                data_path=data_config["path"],
                features=data_config["features"],
                target=data_config["target"],
                numeric_features=data_config["numeric_features"],
                target_values=data_config["target_values"],
                test_size=config["training"]["test_size"],
            )

        logger.debug("PIPELINE | Dataset validation passed")

        split = split_dataset(df, config)
        candidates = []
        for model_name in config["models"]:
            for stage in ("Training", "Evaluation", "Quality Gate"):
                summary[stage] = "SKIPPED"
            with mlflow.start_run(run_name=model_name, nested=True,
                                  tags={"model_type": model_name, "pipeline_run_id": pipeline_run_id}) as child:
                with pipeline_stage(summary, "Training"):
                    training_result = train_model(df=df, config=config, mlflow_run=child,
                                                  model_name=model_name, split=split)
                with pipeline_stage(summary, "Evaluation"):
                    metrics = evaluate_model(training_result["pipeline"], training_result["X_test"],
                                             training_result["y_test"], config["data"]["positive_label"])
                with pipeline_stage(summary, "Quality Gate"):
                    eligible = passes_quality_gate(metrics, config)
                candidates.append({"model_name": model_name, "run_id": child.info.run_id,
                                   "model_uri": training_result["model_uri"], "metrics": metrics,
                                   "eligible": eligible})
                logger.info("COMPARISON | model=%s | accuracy=%.4f | f1=%.4f | eligible=%s",
                            model_name, metrics["accuracy"], metrics["f1_score"], eligible)

        mlflow.log_metrics({
            f"{candidate['model_name']}_{metric}": value
            for candidate in candidates
            for metric, value in candidate["metrics"].items()
        })
        winner = select_best_model(candidates, config)
        mlflow.log_dict({"candidates": candidates, "winner": winner}, "model_comparison.json")
        mlflow.log_params({"selection_metric": config["evaluation"]["selection_metric"],
                           "minimum_accuracy": config["evaluation"]["minimum_accuracy"]})
        summary["Quality Gate"] = "PASSED" if winner else "FAILED"
        if winner is None:
            mlflow.set_tag("quality_gate", "failed")
            return {"status": "rejected", "pipeline_run_id": pipeline_run_id,
                    "run_id": run.info.run_id, "candidates": candidates}

        summary["Accuracy"] = f"{winner['metrics']['accuracy']:.4f}"
        summary["F1"] = f"{winner['metrics']['f1_score']:.4f}"
        summary["Selected Model"] = winner["model_name"]
        mlflow.log_metrics(winner["metrics"])
        mlflow.set_tags({"selected_model": winner["model_name"],
                         "selected_run_id": winner["run_id"], "quality_gate": "passed"})
        with pipeline_stage(summary, "Registration"):
            registered_model = register_model(winner["run_id"], config, model_uri=winner["model_uri"])
        return {"status": "registered", "pipeline_run_id": pipeline_run_id,
                "run_id": run.info.run_id, "selected_run_id": winner["run_id"],
                "selected_model": winner["model_name"], "metrics": winner["metrics"],
                "candidates": candidates, "model_name": registered_model.name,
                "model_version": registered_model.version}


if __name__ == "__main__":
    run_pipeline()
