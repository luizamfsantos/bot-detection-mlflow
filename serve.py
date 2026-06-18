"""FastAPI server that loads an MLflow-logged bot-detection model and exposes a /predict endpoint.

Usage:
    MODEL_URI="runs:/<run_id>/pipeline" python serve.py
    MODEL_URI="runs:/<run_id>/pipeline" uvicorn serve:app --reload
"""

import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_model = None
_model_uri: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_uri
    _model_uri = os.environ.get("MODEL_URI", "")
    if not _model_uri:
        raise RuntimeError(
            "MODEL_URI environment variable is required. Example: runs:/<run_id>/pipeline"
        )
    _model = mlflow.sklearn.load_model(_model_uri)
    yield
    _model = None


app = FastAPI(title="Bot Detection API", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: dict[str, Any]


class PredictResponse(BaseModel):
    prediction: int
    label: str


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": _model_uri}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([request.features])
    prediction = int(_model.predict(df)[0])
    label = "bot" if prediction == 1 else "user"
    return PredictResponse(prediction=prediction, label=label)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
