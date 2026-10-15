"""Train an income classifier on data.csv and track the run with MLflow.

Usage:
    python train.py [--model random_forest|logistic_regression] [--data path/to/data.csv]

Each run validates the data, holds out a test split (saved to data/test.csv for
evaluate.py), logs params, metrics and the fitted pipeline to MLflow, and registers
the pipeline as a new version of config.REGISTERED_MODEL_NAME.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from data_validation import validate
from evaluate import compute_metrics
from logger import get_logger

logger = get_logger("train")

FEATURES = list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)
POSITIVE_CLASS = config.TARGET_CLASSES[1]  # ">50K"


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # Floats rather than ints so the logged MLflow signature tolerates missing values
    X = df[FEATURES].astype({col: "float64" for col in config.NUMERIC_FEATURES})
    y = (df[config.TARGET] == POSITIVE_CLASS).astype(int)
    logger.info("Loaded %d rows from %s (positive rate %.3f)", len(df), path, y.mean())
    return X, y


def build_pipeline(model_name: str) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), list(config.NUMERIC_FEATURES)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(config.CATEGORICAL_FEATURES)),
    ])
    params = config.MODEL_PARAMS[model_name]
    if model_name == "logistic_regression":
        estimator = LogisticRegression(random_state=config.RANDOM_STATE, **params)
    else:
        estimator = RandomForestClassifier(random_state=config.RANDOM_STATE, **params)
    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def train(model_name: str, data_path: Path) -> str:
    logger.info("Validating %s", data_path)
    if not validate(data_path):
        raise SystemExit("Data validation failed; aborting training")

    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y,
    )
    logger.info("Split: %d train rows, %d test rows", len(X_train), len(X_test))

    test_df = X_test.assign(**{config.TARGET: _labels(y_test)})
    test_df.to_csv(config.TEST_DATA_PATH, index=False)
    logger.info("Saved test split to %s", config.TEST_DATA_PATH)

    config.setup_mlflow()

    with mlflow.start_run(run_name=f"train-{model_name}") as run:
        logger.info("MLflow run %s started", run.info.run_id)
        mlflow.set_tag("stage", "train")
        mlflow.log_params({
            "model": model_name,
            "test_size": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE,
            "n_train": len(X_train),
            "n_test": len(X_test),
            **config.MODEL_PARAMS[model_name],
        })
        mlflow.log_input(mlflow.data.from_pandas(X_train.assign(**{config.TARGET: _labels(y_train)}),
                                                 source=data_path.resolve().as_uri(), name="adult-train"),
                         context="training")

        pipeline = build_pipeline(model_name)
        logger.info("Fitting %s", model_name)
        pipeline.fit(X_train, y_train)

        train_metrics = compute_metrics(y_train, pipeline.predict_proba(X_train)[:, 1])
        test_metrics = compute_metrics(y_test, pipeline.predict_proba(X_test)[:, 1])
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        logger.info("Train metrics: %s", _fmt(train_metrics))
        logger.info("Test metrics:  %s", _fmt(test_metrics))

        signature = infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1])
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=config.REGISTERED_MODEL_NAME,
            # MLflow saves sklearn models with skops, which refuses tree internals unless trusted
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        version = str(model_info.registered_model_version)
        logger.info("Registered %s version %s", config.REGISTERED_MODEL_NAME, version)

    logger.info("Next: python evaluate.py --version %s", version)
    return version


def _labels(y: pd.Series) -> pd.Series:
    return y.map({0: config.TARGET_CLASSES[0], 1: POSITIVE_CLASS})


def _fmt(metrics: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(config.MODEL_PARAMS), default=config.DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    args = parser.parse_args()
    train(args.model, args.data)
