"""Inference utilities for NASA C-MAPSS RUL regression models.

Provides model loading, preprocessing alignment, sequence prediction,
and engine health classification compatible with the FastAPI backend.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.data import (
    DEFAULT_SUBSET,
    ROOT,
    latest_windows,
    load_subset,
    make_window,
    validate_subset,
)
from src.model import RULLSTM

ARTIFACTS_DIR = ROOT / "artifacts"


def load_model_bundle(
    subset: str = DEFAULT_SUBSET,
    artifacts_dir: Path = ARTIFACTS_DIR,
    device: str | torch.device = "cpu",
) -> tuple[RULLSTM, Any, dict[str, Any]]:
    """Load model weights, scaler, and configuration for a given subset.

    Args:
        subset: C-MAPSS subset identifier (e.g., 'FD003').
        artifacts_dir: Path to directory containing saved artifacts.
        device: Device to load the PyTorch model onto.

    Returns:
        tuple of (RULLSTM model in eval mode, fitted scaler, configuration dict).
    """
    subset = validate_subset(subset)
    model_path = artifacts_dir / f"model_{subset.lower()}.pt"
    scaler_path = artifacts_dir / f"scaler_{subset.lower()}.joblib"
    config_path = artifacts_dir / f"config_{subset.lower()}.json"

    missing = [p.name for p in (model_path, scaler_path, config_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required model artifacts for {subset}: {', '.join(missing)}. "
            f"Run training first: python -m src.train --subset {subset}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    scaler = joblib.load(scaler_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    features = config.get("features", [])
    if not features or checkpoint["input_size"] != len(features):
        raise ValueError(
            f"Saved checkpoint input size ({checkpoint['input_size']}) does not match "
            f"feature count ({len(features)})."
        )

    model = RULLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, scaler, config


def determine_health_status(
    predicted_rul: float,
    critical_threshold: int = 20,
    warning_threshold: int = 50,
) -> str:
    """Classify engine degradation status based on predicted RUL and thresholds."""
    if predicted_rul <= critical_threshold:
        return "Critical"
    if predicted_rul <= warning_threshold:
        return "Warning"
    return "Normal"


def predict_rul(
    sequences: np.ndarray,
    model: RULLSTM,
    device: str | torch.device = "cpu",
    clip_min: float | None = 0.0,
    clip_max: float | None = None,
) -> np.ndarray:
    """Predict RUL values for 3D numpy sequences of shape (N, sequence_length, features).

    Args:
        sequences: Preprocessed, scaled 3D array of shape (N, seq_len, num_features).
        model: Trained RULLSTM model.
        device: Device on which to run tensor operations.
        clip_min: Minimum RUL clipping bound (default 0.0).
        clip_max: Optional maximum RUL clipping bound.

    Returns:
        1D numpy array of predicted RUL values.
    """
    if sequences.ndim != 3:
        raise ValueError(f"Expected 3D sequences (batch, seq_len, features), got ndim={sequences.ndim}")

    tensor = torch.from_numpy(sequences.astype(np.float32)).to(device)
    with torch.inference_mode():
        preds = model(tensor).cpu().numpy()

    if clip_min is not None or clip_max is not None:
        preds = np.clip(preds, clip_min, clip_max)

    return preds


def predict_engine(
    engine_id: int,
    subset: str = DEFAULT_SUBSET,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, Any]:
    """Generate RUL prediction and status for a single test engine."""
    subset = validate_subset(subset)
    _, test_frame, _ = load_subset(subset)

    group = test_frame[test_frame["engine_id"] == engine_id].sort_values("cycle")
    if group.empty:
        raise ValueError(f"Engine {engine_id} not found in test dataset {subset}.")

    model, scaler, config = load_model_bundle(subset, artifacts_dir=artifacts_dir)
    sequences, engine_ids, last_cycles, _ = latest_windows(
        group, scaler, config["features"], config["sequence_length"]
    )
    predicted_rul = float(predict_rul(sequences, model)[0])
    status = determine_health_status(
        predicted_rul,
        config.get("critical_threshold", 20),
        config.get("warning_threshold", 50),
    )

    return {
        "subset": subset,
        "engine_id": engine_ids[0],
        "last_observed_cycle": last_cycles[0],
        "predicted_rul_cycles": round(predicted_rul, 2),
        "status": status,
        "thresholds": {
            "critical_cycles_or_less": config.get("critical_threshold", 20),
            "warning_cycles_or_less": config.get("warning_threshold", 50),
        },
        "model_type": config.get("model_type", "LSTM"),
        "sequence_length": config.get("sequence_length", 30),
    }
