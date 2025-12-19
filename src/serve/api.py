from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.serve.metrics import http_request_duration_seconds, http_requests_total
from src.settings import settings

app = FastAPI(title="NL Forecasting API", version="0.1.0")


@app.middleware("http")
async def prom_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method
    with http_request_duration_seconds.labels(method=method, path=path).time():
        response = await call_next(request)
    http_requests_total.labels(method=method, path=path, status=str(response.status_code)).inc()
    return response


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/forecast")
def forecast(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Starter endpoint.

    Expected payload (example):
    {
      "series": [{"ds": "2025-01-01T00:00:00", "y": 5.1, "hour_sin": ..., ...}, ...],
      "h": 24
    }

    For a production-ready API:
    - load a model artifact from /models (or from MLflow registry)
    - validate covariates
    - return prediction intervals
    """
    h = int(payload.get("h", settings.horizon_hours))
    series = payload.get("series", [])
    if not series:
        return {"error": "payload.series is required"}
    df = pd.DataFrame(series)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # naive forecast: repeat last value
    last_val = df["y"].dropna().iloc[-1] if df["y"].notna().any() else 0.0
    last = float(last_val)
    future = pd.date_range(df["ds"].iloc[-1], periods=h+1, freq=settings.freq)[1:]
    yhat = [last] * h
    return {
        "model": "naive_api_stub",
        "h": h,
        "forecast": [{"ds": str(ds), "yhat": float(v)} for ds, v in zip(future, yhat)],
    }
