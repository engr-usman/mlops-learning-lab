import logging
import os
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(
    config_path="configs/config.yaml",
):

    path = PROJECT_ROOT / config_path

    logger.info("CONFIG | Loading configuration | path=%s", path)

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        config["mlflow"]["tracking_uri"] = tracking_uri
        logger.info("CONFIG | Using MLflow tracking URI from environment")

    logger.info("CONFIG | Configuration loaded")
    return config
