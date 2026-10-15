"""Train a term-deposit classifier on data.csv and track the run with MLflow.

Usage:
    python train.py [--model logistic_regression|random_forest] [--data path/to/data.csv]
                    [--test-out path/to/test.csv] [--version-out path/to/version.txt]

Each run validates the data, holds out a stratified test split (saved to data/test.csv
for evaluate.py), logs params, metrics and the fitted pipeline to MLflow, and registers
the pipeline as a new version of the registered model named in config.yaml.
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

from config_loader import load_config, setup_mlflow
from data_validation import validate
from evaluate import compute_metrics
from logger import get_logger

config = load_config()
logger = get_logger("train")

TARGET = config["target"]["name"]
TARGET_CLASSES = config["target"]["classes"]
POSITIVE_CLASS = TARGET_CLASSES[1]  # "yes"
DROP_FEATURES = config["training"]["drop_features"]
NUMERIC_FEATURES = [c for c in config["numeric_features"] if c not in DROP_FEATURES]
CATEGORICAL_FEATURES = [c for c in config["categorical_features"] if c not in DROP_FEATURES]
MODELS = {"logistic_regression": LogisticRegression, "random_forest": RandomForestClassifier}


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # Floats rather than ints so the logged MLflow signature tolerates missing values
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].astype({col: "float64" for col in NUMERIC_FEATURES})
    y = (df[TARGET] == POSITIVE_CLASS).astype(int)
    logger.info("Loaded %d rows from %s (positive rate %.3f)", len(df), path, y.mean())
    return X, y


def build_pipeline(model_name: str) -> Pipeline:
    # Preprocessing lives inside the pipeline, so the saved model applies the exact
    # same scaling and encoding to new data at prediction time.
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = MODELS[model_name](
        **config["training"]["models"][model_name],
        random_state=config["split"]["random_state"],  # same seed, same model every run
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train(model_name: str, data_path: Path, test_path: Path) -> str:
    logger.info("Validating %s", data_path)
    if not validate(data_path):
        raise SystemExit("Data validation failed; aborting training")

    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,  # keep the same yes/no ratio in both splits
    )
    logger.info("Split: %d train rows, %d test rows; dropped features: %s",
                len(X_train), len(X_test), DROP_FEATURES)

    test_path.parent.mkdir(parents=True, exist_ok=True)
    X_test.assign(**{TARGET: _labels(y_test)}).to_csv(test_path, index=False)
    logger.info("Saved test split to %s", test_path)

    setup_mlflow()
    model_params = config["training"]["models"][model_name]

    with mlflow.start_run(run_name=f"train-{model_name}") as run:
        logger.info("MLflow run %s started", run.info.run_id)
        mlflow.set_tag("stage", "train")
        mlflow.log_params({
            "model": model_name,
            "test_size": config["split"]["test_size"],
            "random_state": config["split"]["random_state"],
            "dropped_features": ",".join(DROP_FEATURES),
            "n_train": len(X_train),
            "n_test": len(X_test),
            **model_params,
        })
        mlflow.log_input(mlflow.data.from_pandas(X_train.assign(**{TARGET: _labels(y_train)}),
                                                 source=data_path.resolve().as_uri(), name="bank-train"),
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
            registered_model_name=config["mlflow"]["registered_model_name"],
            # MLflow saves sklearn models with skops, which refuses tree internals unless trusted
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        version = str(model_info.registered_model_version)
        logger.info("Registered %s version %s", config["mlflow"]["registered_model_name"], version)

    logger.info("Next: python evaluate.py --version %s", version)
    return version


def _labels(y: pd.Series) -> pd.Series:
    return y.map({0: TARGET_CLASSES[0], 1: POSITIVE_CLASS})


def _fmt(metrics: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), default=config["training"]["model"])
    parser.add_argument("--data", type=Path, default=config["paths"]["data_path"])
    parser.add_argument("--test-out", type=Path, default=config["paths"]["test_data_path"],
                        help="where to save the held-out test split")
    parser.add_argument("--version-out", type=Path,
                        help="also write the registered model version to this file (used by Kubeflow)")
    args = parser.parse_args()
    version = train(args.model, args.data, args.test_out)
    if args.version_out:
        args.version_out.parent.mkdir(parents=True, exist_ok=True)
        args.version_out.write_text(version)
