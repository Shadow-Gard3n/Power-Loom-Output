"""FastAPI prediction service for MP212 - Power Loom Output (Kochi).

Endpoints
    GET  /health   -> {"status": "ok", "model_loaded": bool}
    POST /predict  -> {"prediction": <metres per shift>}

The client's two rules live in the request schema, so a bad request gets
status 422 and never reaches the model:
    1. power_cuts_min above 250 is impossible  -> le=250
    2. loom_age_yr below 0 is a typing mistake -> ge=0

The model is the @champion pipeline saved by the model lead as
models/model.joblib. It must accept a pandas DataFrame with the eight
columns below (preprocessing included inside the pipeline).
"""
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "model.joblib"))

FEATURES = [
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours",
    "fabric",
    "shed",
]


# ---------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger() -> logging.Logger:
    logger = logging.getLogger("mp212")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)  # stdout, never a file
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = get_logger()


# ----------------------------------------------------------------- schema
class PredictRequest(BaseModel):
    """Exactly the eight fields of PROBLEM.md section 5."""

    loom_age_yr: float = Field(..., ge=0, description="age of the loom, years (>= 0)")
    thread_count: int = Field(..., gt=0, description="threads per inch")
    power_cuts_min: float = Field(..., ge=0, le=250, description="power cuts, minutes (<= 250)")
    weaver_exp_yr: float = Field(..., ge=0, description="weaver experience, years")
    humidity_pct: float = Field(..., ge=0, le=100, description="shed humidity, percent")
    shift_hours: float = Field(..., gt=0, le=24, description="shift length, hours")
    fabric: Literal["cotton", "silk", "polyester"]
    shed: Literal["Shed 1", "Shed 2", "Shed 3"]


class PredictResponse(BaseModel):
    prediction: float = Field(..., description="predicted metres_per_shift")


# -------------------------------------------------------------------- app
def load_model(path: Path = MODEL_PATH):
    if not Path(path).exists():
        logger.warning("model file not found", extra={"fields": {"path": str(path)}})
        return None
    model = joblib.load(path)
    logger.info("model loaded", extra={"fields": {"path": str(path)}})
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(title="MP212 Power Loom Output API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    start = time.perf_counter()
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    prediction = float(model.predict(row)[0])
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    # one JSON line per prediction; only machine settings, no personal data
    logger.info(
        "prediction",
        extra={"fields": {"input": req.model_dump(), "prediction": prediction, "latency_ms": latency_ms}},
    )
    return PredictResponse(prediction=prediction)
