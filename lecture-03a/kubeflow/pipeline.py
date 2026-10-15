"""Kubeflow pipeline: ingest -> validate -> train -> evaluate.

Each step runs the project image (kubeflow/Dockerfile) and calls the same scripts you
run locally; KFP passes the files between steps. Train and evaluate log to the MLflow
server from kubeflow/mlflow.yaml, and evaluate promotes the champion like it does locally.

Usage:
    KFP_IMAGE=docker.io/<your-dockerhub-user>/bank-marketing-kfp:v1 python kubeflow/pipeline.py

This writes kubeflow/bank_marketing_pipeline.yaml; upload it in the Kubeflow UI under
Pipelines > Upload pipeline, then click Create run.
"""

import os
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Output, OutputPath

IMAGE = os.environ.get("KFP_IMAGE", "docker.io/YOUR_DOCKERHUB_USER/bank-marketing-kfp:v1")
MLFLOW_TRACKING_URI = os.environ.get("KFP_MLFLOW_URI", "http://mlflow.mlflow.svc.cluster.local:5000")
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
OUTPUT_FILE = Path(__file__).with_name("bank_marketing_pipeline.yaml")


@dsl.container_component
def ingest(source: str, data: Output[Dataset]):
    return dsl.ContainerSpec(image=IMAGE, command=["python", "ingest.py"],
                             args=["--source", source, "--out", data.path])


@dsl.container_component
def validate(data: Input[Dataset]):
    # data_validation.py exits 1 on any failed check, which fails the run here
    return dsl.ContainerSpec(image=IMAGE, command=["python", "data_validation.py"], args=[data.path])


@dsl.container_component
def train(model: str, data: Input[Dataset], test_data: Output[Dataset], version: OutputPath(str)):
    return dsl.ContainerSpec(image=IMAGE, command=["python", "train.py"],
                             args=["--model", model, "--data", data.path,
                                   "--test-out", test_data.path, "--version-out", version])


@dsl.container_component
def evaluate(version: str, test_data: Input[Dataset]):
    return dsl.ContainerSpec(image=IMAGE, command=["python", "evaluate.py"],
                             args=["--version", version, "--data", test_data.path])


@dsl.pipeline(name="bank-marketing", description="Ingest, validate, train and evaluate the term-deposit classifier")
def bank_marketing_pipeline(model: str = "random_forest", source: str = UCI_URL):
    ingest_task = ingest(source=source)
    validate_task = validate(data=ingest_task.outputs["data"])

    train_task = train(model=model, data=ingest_task.outputs["data"]).after(validate_task)
    evaluate_task = evaluate(version=train_task.outputs["version"],
                             test_data=train_task.outputs["test_data"])

    for task in (train_task, evaluate_task):
        task.set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
        task.set_caching_options(False)  # every run should register and evaluate a new version


if __name__ == "__main__":
    compiler.Compiler().compile(bank_marketing_pipeline, str(OUTPUT_FILE))
    print(f"Wrote {OUTPUT_FILE} (image: {IMAGE})")
