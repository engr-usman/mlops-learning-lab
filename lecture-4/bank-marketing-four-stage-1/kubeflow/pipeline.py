"""Four-stage Bank Marketing pipeline for Kubeflow Pipelines v2.

validate-dataset -> train-candidates -> evaluate-candidates -> register-winner

Each step runs `python -m src.component_cli <stage>` in the image built from
kubeflow/Dockerfile and hands typed artifacts to the next step.

Compile (from the project root):
    KFP_IMAGE=docker.io/<your-dockerhub-user>/bank-marketing-kfp-four-stage:v1 \
        python kubeflow/pipeline.py
"""

import os
import sys
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Output

IMAGE = os.getenv("KFP_IMAGE", "")
COMMAND = ["python", "-m", "src.component_cli"]
EXECUTOR_INPUT = dsl.PIPELINE_TASK_EXECUTOR_INPUT_PLACEHOLDER
DEFAULT_TRACKING_URI = "http://mlflow.mlflow-four-stage.svc.cluster.local:5000"


@dsl.container_component
def validate_dataset(dataset_path: str, config_path: str,
                     validated_dataset: Output[Dataset], configuration: Output[Artifact],
                     validation_report: Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE, command=COMMAND, args=[
        "validate", "--dataset", dataset_path, "--config", config_path,
        "--validated", validated_dataset.path, "--snapshot", configuration.path,
        "--validation", validation_report.path, "--executor-input", EXECUTOR_INPUT,
    ])


@dsl.container_component
def train_candidates(validated_dataset: Input[Dataset], configuration: Input[Artifact],
                     validation_report: Input[Artifact], tracking_uri: str, execution_id: str,
                     candidate_manifest: Output[Artifact], heldout_dataset: Output[Dataset],
                     split_metadata: Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE, command=COMMAND, args=[
        "train", "--validated", validated_dataset.path, "--snapshot", configuration.path,
        "--validation", validation_report.path, "--tracking-uri", tracking_uri,
        "--execution-id", execution_id, "--manifest", candidate_manifest.path,
        "--heldout", heldout_dataset.path, "--split", split_metadata.path,
        "--executor-input", EXECUTOR_INPUT,
    ])


@dsl.container_component
def evaluate_candidates(candidate_manifest: Input[Artifact], heldout_dataset: Input[Dataset],
                        split_metadata: Input[Artifact], configuration: Input[Artifact], tracking_uri: str,
                        evaluation_report: Output[Artifact], evaluation_metrics: Output[Metrics]):
    return dsl.ContainerSpec(image=IMAGE, command=COMMAND, args=[
        "evaluate", "--manifest", candidate_manifest.path, "--heldout", heldout_dataset.path,
        "--split", split_metadata.path, "--snapshot", configuration.path,
        "--tracking-uri", tracking_uri, "--evaluation", evaluation_report.path,
        "--metrics", evaluation_metrics.path, "--executor-input", EXECUTOR_INPUT,
    ])


@dsl.container_component
def register_winner(evaluation_report: Input[Artifact], configuration: Input[Artifact], tracking_uri: str,
                    registration_result: Output[Artifact]):
    return dsl.ContainerSpec(image=IMAGE, command=COMMAND, args=[
        "register", "--evaluation", evaluation_report.path, "--snapshot", configuration.path,
        "--tracking-uri", tracking_uri, "--registration", registration_result.path,
        "--executor-input", EXECUTOR_INPUT,
    ])


def configure(task: dsl.PipelineTask, cpu: str, cpu_limit: str, memory: str, memory_limit: str) -> None:
    """Set resources, disable caching and pull the image only when the node lacks it."""
    task.set_caching_options(False)
    task.set_cpu_request(cpu).set_cpu_limit(cpu_limit)
    task.set_memory_request(memory).set_memory_limit(memory_limit)
    # Same as kfp-kubernetes set_image_pull_policy, without the extra dependency.
    task.platform_config["kubernetes"] = {"imagePullPolicy": "IfNotPresent"}


@dsl.pipeline(name="bank-marketing-four-stage",
              description="Validate, train, evaluate and register the Bank Marketing model in four steps.")
def bank_marketing_pipeline(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    dataset_path: str = "/app/data/balance.csv",
    config_path: str = "/app/configs/config.yaml",
):
    validation = validate_dataset(dataset_path=dataset_path, config_path=config_path)
    training = train_candidates(
        validated_dataset=validation.outputs["validated_dataset"],
        configuration=validation.outputs["configuration"],
        validation_report=validation.outputs["validation_report"],
        tracking_uri=tracking_uri, execution_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    )
    evaluation = evaluate_candidates(
        candidate_manifest=training.outputs["candidate_manifest"],
        heldout_dataset=training.outputs["heldout_dataset"],
        split_metadata=training.outputs["split_metadata"],
        configuration=validation.outputs["configuration"], tracking_uri=tracking_uri,
    )
    registration = register_winner(
        evaluation_report=evaluation.outputs["evaluation_report"],
        configuration=validation.outputs["configuration"], tracking_uri=tracking_uri,
    )
    validation.set_display_name("validate-dataset")
    training.set_display_name("train-candidates")
    evaluation.set_display_name("evaluate-candidates")
    registration.set_display_name("register-winner")
    configure(validation, "100m", "1", "256Mi", "1Gi")
    configure(training, "500m", "2", "1Gi", "3Gi")
    configure(evaluation, "250m", "1", "512Mi", "2Gi")
    configure(registration, "100m", "1", "256Mi", "1Gi")


if __name__ == "__main__":
    if not IMAGE:
        sys.exit("Set KFP_IMAGE, e.g. KFP_IMAGE=docker.io/<user>/bank-marketing-kfp-four-stage:v1")
    output = Path(__file__).parent / "bank_marketing_pipeline.yaml"
    compiler.Compiler().compile(bank_marketing_pipeline, str(output))
    print(f"Compiled {output} | image={IMAGE}")
