"""Pydantic schemas for the Predictive Maintenance API."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status ('ok' or 'degraded')")
    default_subset: str = Field(..., description="Default C-MAPSS subset identifier")
    dataset_ready: bool = Field(..., description="True if required data files are present")
    model_ready: bool = Field(..., description="True if trained model bundle is available")
    model_type: str = Field("LSTM", description="Architecture of the predictive model")


class SubsetInfo(BaseModel):
    subset: str
    dataset_available: bool
    model_ready: bool


class SubsetsResponse(BaseModel):
    default_subset: str
    subsets: list[SubsetInfo]


class EngineSummaryItem(BaseModel):
    engine_id: int
    last_observed_cycle: int | None = None
    predicted_rul_cycles: float | None = None
    status: str | None = None


class EnginesResponse(BaseModel):
    subset: str
    engine_ids: list[int]
    count: int
    attention_count: int | None = None
    engines: list[EngineSummaryItem]


class EngineDetailResponse(BaseModel):
    subset: str
    engine_id: int
    last_observed_cycle: int
    reference_rul_cycles: float
    total_cycles: int
    sensor_names: list[str]
    setting_names: list[str] = ["setting_1", "setting_2", "setting_3"]
    cycles: list[dict[str, Any]]


class PredictionThresholds(BaseModel):
    critical_cycles_or_less: int = 20
    warning_cycles_or_less: int = 50


class PredictionResponse(BaseModel):
    subset: str
    engine_id: int | None = None
    last_observed_cycle: int
    predicted_rul_cycles: float
    status: str = Field(..., description="'Critical', 'Warning', or 'Normal'")
    thresholds: PredictionThresholds
    model_type: str = "LSTM"
    sequence_length: int = 30
    note: str = "Benchmark estimate on simulated data only; not a maintenance recommendation."


class PredictRequest(BaseModel):
    subset: str = "FD003"
    engine_id: int | None = Field(None, description="Preloaded test engine ID (1-indexed)")
    sequence: list[list[float]] | None = Field(
        None,
        description="Optional custom 2D sequence of shape (time_steps, features) with matching feature ordering",
    )
    critical_threshold: int | None = Field(None, description="Override critical threshold (default 20)")
    warning_threshold: int | None = Field(None, description="Override warning threshold (default 50)")


class MetricsResponse(BaseModel):
    subset: str
    model_type: str
    device: str
    best_epoch: int
    epochs_completed: int | None = None
    validation_engines: int | None = None
    validation_mae: float | None = None
    validation_rmse: float | None = None
    test_engines: int
    test_mae: float
    test_rmse: float
    test_nasa_score: float
    test_r2_score: float | None = None
    sequence_length: int
    feature_count: int
    rul_cap: int | None = None
    baselines: dict[str, Any] | None = None
