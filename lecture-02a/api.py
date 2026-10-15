"""Serve the champion model with FastAPI.

Usage:
    uvicorn api:app --reload

Loads models:/<REGISTERED_MODEL_NAME>@<CHAMPION_ALIAS> from the MLflow registry at
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

import config
from logger import get_logger

logger = get_logger("api")

FEATURES = list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)
state: dict = {}


def _choices(column: str):
    return Literal[tuple(config.CATEGORICAL_FEATURES[column])]


class Person(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"example": {
            "age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors",
            "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical",
            "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 2174,
            "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States",
        }},
    )

    age: int = Field(ge=config.NUMERIC_FEATURES["age"][0], le=config.NUMERIC_FEATURES["age"][1])
    workclass: _choices("workclass")
    fnlwgt: int = Field(gt=0)
    education: _choices("education")
    education_num: int = Field(alias="education-num", ge=1, le=16)
    marital_status: _choices("marital-status") = Field(alias="marital-status")
    occupation: _choices("occupation")
    relationship: _choices("relationship")
    race: _choices("race")
    sex: _choices("sex")
    capital_gain: int = Field(alias="capital-gain", ge=0)
    capital_loss: int = Field(alias="capital-loss", ge=0)
    hours_per_week: int = Field(alias="hours-per-week", ge=1, le=99)
    native_country: _choices("native-country") = Field(alias="native-country")


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    income: Literal[tuple(config.TARGET_CLASSES)]
    probability_over_50k: float
    model_version: str


class BatchRequest(BaseModel):
    instances: list[Person] = Field(min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    predictions: list[Prediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    uri = f"models:/{config.REGISTERED_MODEL_NAME}@{config.CHAMPION_ALIAS}"
    try:
        version = MlflowClient().get_model_version_by_alias(config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS)
        state["model"] = mlflow.sklearn.load_model(uri)
        state["version"] = str(version.version)
        logger.info("Loaded %s (version %s)", uri, state["version"])
    except Exception:
        logger.exception("Could not load %s; run train.py and evaluate.py first", uri)
    yield
    state.clear()


app = FastAPI(title="Adult Income Classifier", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


def _predict(people: list[Person]) -> list[Prediction]:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = pd.DataFrame([p.model_dump(by_alias=True) for p in people])[FEATURES]
    proba = state["model"].predict_proba(X)[:, 1]
    return [
        Prediction(
            income=config.TARGET_CLASSES[int(p >= 0.5)],
            probability_over_50k=round(float(p), 4),
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
        "name": config.REGISTERED_MODEL_NAME,
        "alias": config.CHAMPION_ALIAS,
        "version": state["version"],
        "features": FEATURES,
    }


@app.post("/predict", response_model=Prediction)
def predict(person: Person):
    prediction = _predict([person])[0]
    logger.info("Prediction: %s (p=%.4f)", prediction.income, prediction.probability_over_50k)
    return prediction


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(request: BatchRequest):
    predictions = _predict(request.instances)
    logger.info("Batch prediction for %d instances", len(predictions))
    return BatchPrediction(predictions=predictions)
