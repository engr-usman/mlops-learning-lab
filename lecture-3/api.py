"""Serve the champion model with FastAPI.

Usage:
    uvicorn api:app --reload

Loads models:/<registered_model_name>@<champion_alias> from the MLflow registry at
startup. Interactive docs are at http://127.0.0.1:8000/docs.
"""

import time
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from config_loader import load_config, setup_mlflow
from logger import get_logger

config = load_config()
logger = get_logger("api")

TARGET_CLASSES = config["target"]["classes"]
MODEL_NAME = config["mlflow"]["registered_model_name"]
CHAMPION_ALIAS = config["mlflow"]["champion_alias"]
DROP_FEATURES = config["training"]["drop_features"]
NUMERIC_FEATURES = [c for c in config["numeric_features"] if c not in DROP_FEATURES]
CATEGORICAL_FEATURES = [c for c in config["categorical_features"] if c not in DROP_FEATURES]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
state: dict = {}


def _choices(column: str):
    return Literal[tuple(config["categorical_features"][column])]


def _range(column: str):
    low, high = config["numeric_features"][column]
    return Field(ge=low, le=high)


class Customer(BaseModel):
    """One client, with the same columns as data.csv minus the target and dropped features."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 58, "job": "management", "marital": "married", "education": "tertiary",
        "default": "no", "balance": 2143, "housing": "yes", "loan": "no", "contact": "unknown",
        "day": 5, "month": "may", "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown",
    }})

    age: int = _range("age")
    job: _choices("job")
    marital: _choices("marital")
    education: _choices("education")
    default: _choices("default")
    balance: int = _range("balance")
    housing: _choices("housing")
    loan: _choices("loan")
    contact: _choices("contact")
    day: int = _range("day")
    month: _choices("month")
    campaign: int = _range("campaign")
    pdays: int = _range("pdays")
    previous: int = _range("previous")
    poutcome: _choices("poutcome")


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    subscribed: Literal[tuple(TARGET_CLASSES)]
    probability_yes: float
    model_version: str


class BatchRequest(BaseModel):
    instances: list[Customer] = Field(min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    predictions: list[Prediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_mlflow()
    uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    try:
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        state["model"] = mlflow.sklearn.load_model(uri)
        state["version"] = str(version.version)
        logger.info("Loaded %s (version %s)", uri, state["version"])
    except Exception:
        logger.exception("Could not load %s; run train.py and evaluate.py first", uri)
    yield
    state.clear()


app = FastAPI(title="Bank Marketing Classifier", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


def _predict(customers: list[Customer]) -> list[Prediction]:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    X = X.astype({col: "float64" for col in NUMERIC_FEATURES})  # match the training signature
    proba = state["model"].predict_proba(X)[:, 1]
    return [
        Prediction(
            subscribed=TARGET_CLASSES[int(p >= 0.5)],
            probability_yes=round(float(p), 4),
            model_version=state["version"],
        )
        for p in proba
    ]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "name": MODEL_NAME,
        "alias": CHAMPION_ALIAS,
        "version": state["version"],
        "features": FEATURES,
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    prediction = _predict([customer])[0]
    logger.info("Prediction: %s (p=%.4f)", prediction.subscribed, prediction.probability_yes)
    return prediction


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(request: BatchRequest):
    predictions = _predict(request.instances)
    logger.info("Batch prediction for %d instances", len(predictions))
    return BatchPrediction(predictions=predictions)
