"""Load project settings from config.yaml.

The file defaults to config.yaml in the project root. Set the CONFIG_PATH environment
variable to use a different file, e.g. one mounted into a Docker container.
Relative paths under `paths:` are resolved against the project root.
"""

import os
from functools import cache
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


@cache
def load_config() -> dict:
    config_path = Path(os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH))
    with config_path.open() as f:
        config = yaml.safe_load(f)
    config["paths"] = {name: ROOT_DIR / path for name, path in config["paths"].items()}
    return config


def setup_mlflow() -> None:
    """Point MLflow at the tracking store and experiment, wherever the script runs from.

    Locally this is the project's sqlite DB. Set MLFLOW_TRACKING_URI (docker-compose does)
    to use an MLflow server instead; the server then stores the model files itself.
    """
    import mlflow

    config = load_config()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        artifact_location = None  # the server decides where model files go
    else:
        mlflow.set_tracking_uri(f"sqlite:///{config['paths']['mlflow_db']}")
        artifact_location = config["paths"]["mlflow_artifacts"].as_uri()

    name = config["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)
