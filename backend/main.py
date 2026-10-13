"""FastAPI application for NASA C-MAPSS Predictive Maintenance RUL Inference."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Path as PathParameter, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    EngineDetailResponse,
    EnginesResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    PredictRequest,
    SubsetsResponse,
)
from src.data import (
    DEFAULT_SUBSET,
    ROOT,
    SUBSETS,
    latest_windows,
    load_subset,
    make_window,
    to_records,
    validate_subset,
)
from src.inference import determine_health_status, load_model_bundle, predict_rul

ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

app = FastAPI(
    title="Deep Learning-Based Predictive Maintenance API",
    version="2.0.0",
    description=(
        "RUL estimates for NASA C-MAPSS simulated turbofan engines using PyTorch LSTM. "
        "Benchmark research only; not for real aircraft maintenance decisions."
    ),
)

# CORS configuration for Vite / Next / React local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _artifact_path(prefix: str, subset: str, suffix: str) -> Path:
    return ARTIFACTS / f"{prefix}_{subset.lower()}.{suffix}"


def _read_dataset(subset: str):
    try:
        return load_subset(subset)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@lru_cache(maxsize=len(SUBSETS))
def _get_bundle(subset: str):
    """Cached loader for model bundle artifacts."""
    try:
        return load_model_bundle(subset=subset, artifacts_dir=ARTIFACTS)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained {subset} model bundle is available. "
                f"Train it first with: python -m src.train --subset {subset}"
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"The saved {subset} model bundle could not be loaded: {exc}",
        ) from exc


@app.get("/health", response_model=HealthResponse, summary="Check API, dataset, and model readiness")
@app.get("/api/health", response_model=HealthResponse, include_in_schema=False)
def health() -> dict[str, Any]:
    subset = DEFAULT_SUBSET
    model_ready = all(
        _artifact_path(prefix, subset, suffix).is_file()
        for prefix, suffix in (("model", "pt"), ("scaler", "joblib"), ("config", "json"))
    )
    dataset_ready = all(
        (ROOT / "data" / f"{prefix}_{subset}.txt").is_file()
        for prefix in ("train", "test", "RUL")
    )
    return {
        "status": "ok" if dataset_ready else "degraded",
        "default_subset": subset,
        "dataset_ready": dataset_ready,
        "model_ready": model_ready,
        "model_type": "LSTM",
    }


@app.get("/api/subsets", response_model=SubsetsResponse, summary="List supported C-MAPSS subsets and readiness")
def subsets() -> dict[str, Any]:
    return {
        "default_subset": DEFAULT_SUBSET,
        "subsets": [
            {
                "subset": subset,
                "dataset_available": all(
                    (ROOT / "data" / f"{prefix}_{subset}.txt").is_file()
                    for prefix in ("train", "test", "RUL")
                ),
                "model_ready": _artifact_path("model", subset, "pt").is_file()
                and _artifact_path("scaler", subset, "joblib").is_file()
                and _artifact_path("config", subset, "json").is_file(),
            }
            for subset in SUBSETS
        ],
    }


@app.get("/api/metrics", response_model=MetricsResponse, summary="Get validation and official test benchmark metrics")
def metrics(subset: str = Query(DEFAULT_SUBSET, description="Dataset subset identifier")) -> dict[str, Any]:
    try:
        normalized_subset = validate_subset(subset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    metrics_path = _artifact_path("metrics", normalized_subset, "json")
    if not metrics_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation metrics for {normalized_subset} are not available; train the model first.",
        )
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=f"Evaluation metrics could not be read: {exc}") from exc

    # Check for additional baseline metrics from reports
    report_path = REPORTS / f"evaluation_report_{normalized_subset.lower()}.json"
    if report_path.is_file():
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
            if "baselines" in report_data and "baselines" not in data:
                data["baselines"] = report_data["baselines"]
            if "metrics" in report_data and "r2_score" in report_data["metrics"]:
                data["test_r2_score"] = report_data["metrics"]["r2_score"]
        except Exception:
            pass

    return data


@app.get("/api/engines", response_model=EnginesResponse, summary="List official test engines and current predicted status")
def engines(subset: str = Query(DEFAULT_SUBSET, description="Dataset subset identifier")) -> dict[str, Any]:
    try:
        normalized_subset = validate_subset(subset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _, test_frame, _ = _read_dataset(normalized_subset)
    ids = sorted(int(value) for value in test_frame["engine_id"].unique())

    response: dict[str, Any] = {
        "subset": normalized_subset,
        "engine_ids": ids,
        "count": len(ids),
        "attention_count": None,
        "engines": [{"engine_id": engine_id} for engine_id in ids],
    }

    try:
        model, scaler, config = _get_bundle(normalized_subset)
    except HTTPException:
        return response

    sequences, engine_ids, last_cycles, _ = latest_windows(
        test_frame, scaler, config["features"], config["sequence_length"]
    )
    predictions = predict_rul(sequences, model, clip_min=0.0)

    engine_summaries = []
    crit_th = config.get("critical_threshold", 20)
    warn_th = config.get("warning_threshold", 50)

    for engine_id, last_cycle, prediction in zip(engine_ids, last_cycles, predictions, strict=True):
        val = float(prediction)
        status = determine_health_status(val, crit_th, warn_th)
        engine_summaries.append(
            {
                "engine_id": engine_id,
                "last_observed_cycle": last_cycle,
                "predicted_rul_cycles": round(val, 2),
                "status": status,
            }
        )

    response["engines"] = engine_summaries
    response["attention_count"] = sum(item["status"] != "Normal" for item in engine_summaries)
    return response


@app.get("/api/engine/{engine_id}", response_model=EngineDetailResponse, summary="Get a test engine's cycle-by-cycle sensor history")
def engine(
    engine_id: int = PathParameter(..., ge=1, description="Engine identifier"),
    subset: str = Query(DEFAULT_SUBSET, description="Dataset subset identifier"),
) -> dict[str, Any]:
    try:
        normalized_subset = validate_subset(subset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _, test_frame, rul = _read_dataset(normalized_subset)
    group = test_frame[test_frame["engine_id"] == engine_id].sort_values("cycle")
    if group.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} was not found in {normalized_subset}.")

    ids = sorted(int(value) for value in test_frame["engine_id"].unique())
    ref_rul = float(rul.iloc[ids.index(engine_id)])

    return {
        "subset": normalized_subset,
        "engine_id": engine_id,
        "last_observed_cycle": int(group["cycle"].iloc[-1]),
        "reference_rul_cycles": ref_rul,
        "total_cycles": len(group),
        "sensor_names": [column for column in group.columns if column.startswith("s")],
        "setting_names": ["setting_1", "setting_2", "setting_3"],
        "cycles": to_records(group),
    }


@app.get("/api/engine/{engine_id}/predict", response_model=PredictionResponse, summary="Predict an engine's remaining useful life")
def predict_engine(
    engine_id: int = PathParameter(..., ge=1, description="Engine identifier"),
    subset: str = Query(DEFAULT_SUBSET, description="Dataset subset identifier"),
) -> dict[str, Any]:
    try:
        normalized_subset = validate_subset(subset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _, test_frame, _ = _read_dataset(normalized_subset)
    group = test_frame[test_frame["engine_id"] == engine_id].sort_values("cycle")
    if group.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} was not found in {normalized_subset}.")

    model, scaler, config = _get_bundle(normalized_subset)
    sequences, engine_ids, last_cycles, _ = latest_windows(
        group, scaler, config["features"], config["sequence_length"]
    )
    prediction = float(predict_rul(sequences, model, clip_min=0.0)[0])

    crit_th = config.get("critical_threshold", 20)
    warn_th = config.get("warning_threshold", 50)
    status = determine_health_status(prediction, crit_th, warn_th)

    return {
        "subset": normalized_subset,
        "engine_id": engine_ids[0],
        "last_observed_cycle": last_cycles[0],
        "predicted_rul_cycles": round(prediction, 2),
        "status": status,
        "thresholds": {
            "critical_cycles_or_less": crit_th,
            "warning_cycles_or_less": warn_th,
        },
        "model_type": config.get("model_type", "LSTM"),
        "sequence_length": config.get("sequence_length", 30),
        "note": "Benchmark estimate on simulated data only; not a maintenance recommendation.",
    }


@app.post("/api/predict", response_model=PredictionResponse, summary="Predict RUL for an engine or raw sequence")
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        normalized_subset = validate_subset(request.subset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model, scaler, config = _get_bundle(normalized_subset)
    features = config["features"]
    seq_len = config["sequence_length"]
    crit_th = request.critical_threshold if request.critical_threshold is not None else config.get("critical_threshold", 20)
    warn_th = request.warning_threshold if request.warning_threshold is not None else config.get("warning_threshold", 50)

    if crit_th >= warn_th:
        raise HTTPException(status_code=400, detail="critical_threshold must be strictly less than warning_threshold")

    # Path A: Preloaded engine ID
    if request.engine_id is not None:
        _, test_frame, _ = _read_dataset(normalized_subset)
        group = test_frame[test_frame["engine_id"] == request.engine_id].sort_values("cycle")
        if group.empty:
            raise HTTPException(status_code=404, detail=f"Engine {request.engine_id} not found in {normalized_subset}.")

        sequences, engine_ids, last_cycles, _ = latest_windows(
            group, scaler, features, seq_len
        )
        prediction = float(predict_rul(sequences, model, clip_min=0.0)[0])
        status = determine_health_status(prediction, crit_th, warn_th)

        return {
            "subset": normalized_subset,
            "engine_id": engine_ids[0],
            "last_observed_cycle": last_cycles[0],
            "predicted_rul_cycles": round(prediction, 2),
            "status": status,
            "thresholds": {
                "critical_cycles_or_less": crit_th,
                "warning_cycles_or_less": warn_th,
            },
            "model_type": config.get("model_type", "LSTM"),
            "sequence_length": seq_len,
            "note": "Benchmark estimate on simulated data only; not a maintenance recommendation.",
        }

    # Path B: Custom sequence
    if request.sequence is not None:
        raw_arr = np.array(request.sequence, dtype=np.float32)
        if raw_arr.ndim != 2 or raw_arr.shape[0] == 0:
            raise HTTPException(status_code=400, detail="sequence must be a non-empty 2D list of shape (time_steps, features)")
        if raw_arr.shape[1] != len(features):
            raise HTTPException(
                status_code=400,
                detail=f"sequence feature count ({raw_arr.shape[1]}) does not match model expected features ({len(features)}).",
            )

        scaled = scaler.transform(raw_arr)
        window = make_window(scaled, seq_len)
        window_batch = np.expand_dims(window, axis=0)

        prediction = float(predict_rul(window_batch, model, clip_min=0.0)[0])
        status = determine_health_status(prediction, crit_th, warn_th)

        return {
            "subset": normalized_subset,
            "engine_id": None,
            "last_observed_cycle": raw_arr.shape[0],
            "predicted_rul_cycles": round(prediction, 2),
            "status": status,
            "thresholds": {
                "critical_cycles_or_less": crit_th,
                "warning_cycles_or_less": warn_th,
            },
            "model_type": config.get("model_type", "LSTM"),
            "sequence_length": seq_len,
            "note": "Benchmark estimate on simulated data only; not a maintenance recommendation.",
        }

    raise HTTPException(status_code=400, detail="Either engine_id or sequence must be provided in request body.")
