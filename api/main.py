from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model_xgb_tuned.joblib")
METRICS_PATH = Path("reports/metrics_xgb_tuned.json")


app = FastAPI(title="Churn Prediction API", version="0.1.0")

_model = None


class PredictRequest(BaseModel):
    # Example:
    # {
    #   "gender": "Female",
    #   "SeniorCitizen": 0,
    #   "tenure": 12,

    features: Dict[str, Any] = Field(..., description="Feature dictionary")


class PredictResponse(BaseModel):
    churn_probability: float
    churn_label: int
    threshold: float


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Train first.")
        _model = joblib.load(MODEL_PATH)
    return _model

def load_threshold(default: float = 0.5) -> float:
    if not METRICS_PATH.exists():
        return default

    try:
        import json
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return float(metrics.get("best_threshold_f1", default))
    except Exception:
        return default


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/info")
def info():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    clf = load_model()

    X = pd.DataFrame([req.features])

    proba = float(clf.predict_proba(X)[:, 1][0])
    threshold = load_threshold()
    label = int(proba >= threshold)

    return PredictResponse(churn_probability=proba, churn_label=label, threshold=threshold)
