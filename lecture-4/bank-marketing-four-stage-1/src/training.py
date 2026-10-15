import hashlib
import logging
from contextlib import nullcontext

import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)



def calculate_file_hash(file_path):

    logger.debug("TRAINING | Calculating dataset hash | path=%s", file_path)
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    logger.debug("TRAINING | Dataset hash calculated | path=%s", file_path)
    return sha256.hexdigest()


def create_pipeline(config, model_name="decision_tree"):
    features = config["data"]["features"]
    model_config = config["models"][model_name]
    logger.info("TRAINING | Creating pipeline | model=%s", model_name)

    categorical_features = config["data"]["categorical_features"]
    numeric_features = config["data"]["numeric_features"]
    if (len(categorical_features + numeric_features) != len(features)
            or set(categorical_features) & set(numeric_features)
            or set(categorical_features + numeric_features) != set(features)):
        raise ValueError("Categorical and numeric features must partition the configured features.")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                ),
                categorical_features,
            ),
            ("numeric", StandardScaler() if model_name == "logistic_regression" else "passthrough", numeric_features),
        ]
    )

    classifiers = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }
    classifier = classifiers[model_name](
        **model_config, random_state=config["project"]["random_state"]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    logger.debug("TRAINING | Model pipeline created")
    return pipeline


def split_dataset(df, config):
    return train_test_split(
        df[config["data"]["features"]], df[config["data"]["target"]],
        test_size=config["training"]["test_size"],
        random_state=config["project"]["random_state"], stratify=df[config["data"]["target"]],
    )


def train_model(df, config, run_name=None, mlflow_run=None,
                model_name="decision_tree", split=None):
    """Train one candidate using a shared split and the supplied tracking run."""
    X_train, X_test, y_train, y_test = split if split is not None else split_dataset(df, config)

    logger.info(
        "TRAINING | Dataset split | training_rows=%d | test_rows=%d",
        len(X_train),
        len(X_test),
    )
    pipeline = create_pipeline(config, model_name)

    run_context = (
        nullcontext(mlflow_run)
        if mlflow_run is not None
        else mlflow.start_run(run_name=run_name)
    )
    with run_context as run:

        logger.info("TRAINING | Using MLflow run | mlflow_run_id=%s", run.info.run_id)
        logger.info("TRAINING | Fitting model")
        pipeline.fit(
            X_train,
            y_train,
        )
        logger.info("TRAINING | Model fitted")

        data_path = config["data"]["path"]

        dataset_hash = calculate_file_hash(
            data_path
        )
        logger.info("TRAINING | Dataset hashed | dataset_hash=%s", dataset_hash)

        mlflow.log_params({
            "model_type": model_name,
            **{key: str(value) for key, value in config["models"][model_name].items()},
            "random_state": config["project"]["random_state"],
            "test_size": config["training"]["test_size"],
            "training_rows": len(X_train), "test_rows": len(X_test),
            "dataset_hash": dataset_hash,
        })

        mlflow.set_tags(
            {
                "project": config["project"][
                    "name"
                ],
                "pipeline_stage": "training",
            }
        )

        logger.info("TRAINING | Parameters and tags logged")
        logger.info("TRAINING | Saving model to MLflow")
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            # Tree internals belong to classifiers built locally by create_pipeline.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        logger.info("TRAINING | Completed | mlflow_run_id=%s", run.info.run_id)
        return {
            "run_id": run.info.run_id,
            "model_uri": getattr(model_info, "model_uri", None) or f"runs:/{run.info.run_id}/model",
            "model_name": model_name,
            "pipeline": pipeline,
            "X_test": X_test,
            "y_test": y_test,
        }
