from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from actions import execute_intent
from ml_intents import (
    DATASET_PATH,
    ensure_model_artifacts,
    load_metrics,
    predict_text,
    train_and_save_model,
)


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="ML Voice Assistant",
    description="Final year project backend for ML-based intent classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Command text to classify.")


@app.on_event("startup")
def startup_event() -> None:
    ensure_model_artifacts()


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "dataset_available": DATASET_PATH.exists()}


@app.get("/api/model-info")
def model_info() -> dict:
    return load_metrics()


@app.post("/api/train")
def train_model() -> dict:
    try:
        return train_and_save_model()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict:
    prediction = predict_text(request.text)
    action = execute_intent(prediction["intent"], request.text)
    return {
        "prediction": prediction,
        "action": action,
    }
