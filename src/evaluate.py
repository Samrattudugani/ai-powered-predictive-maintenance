"""Evaluation and benchmarking module for NASA C-MAPSS models.

Computes MAE, RMSE, R^2, and official NASA PHM asymmetric score.
Provides side-by-side benchmarking against naive and regression baselines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import (
    DEFAULT_SUBSET,
    ROOT,
    add_train_targets,
    latest_windows,
    load_subset,
    validate_subset,
)
from src.inference import load_model_bundle, predict_rul
from src.train import nasa_score

ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"


def evaluate_baselines(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    test_reference: np.ndarray,
    feature_cols: list[str],
    scaler: Any,
    rul_cap: int = 125,
) -> dict[str, dict[str, float]]:
    """Compute benchmark performance for naive and standard regression baselines."""
    # Compute capped training RUL
    train_with_targets = add_train_targets(train_frame, rul_cap=rul_cap)
    median_rul = float(train_with_targets["rul"].median())
    mean_rul = float(train_with_targets["rul"].mean())

    # Baseline 1: Naive Median
    median_preds = np.full_like(test_reference, fill_value=median_rul)
    median_metrics = {
        "mae": float(mean_absolute_error(test_reference, median_preds)),
        "rmse": float(np.sqrt(mean_squared_error(test_reference, median_preds))),
        "nasa_score": nasa_score(median_preds, test_reference),
        "r2": float(r2_score(test_reference, median_preds)),
    }

    # Baseline 2: Naive Mean
    mean_preds = np.full_like(test_reference, fill_value=mean_rul)
    mean_metrics = {
        "mae": float(mean_absolute_error(test_reference, mean_preds)),
        "rmse": float(np.sqrt(mean_squared_error(test_reference, mean_preds))),
        "nasa_score": nasa_score(mean_preds, test_reference),
        "r2": float(r2_score(test_reference, mean_preds)),
    }

    # Baseline 3: Ridge Regressor on last cycle snapshot
    train_last = train_with_targets.sort_values(["engine_id", "cycle"]).groupby("engine_id").last().reset_index()
    test_last = test_frame.sort_values(["engine_id", "cycle"]).groupby("engine_id").last().reset_index()

    x_train = scaler.transform(train_last[feature_cols].to_numpy(dtype=np.float32))
    y_train = train_last["rul"].to_numpy(dtype=np.float32)
    x_test = scaler.transform(test_last[feature_cols].to_numpy(dtype=np.float32))

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(x_train, y_train)
    ridge_preds = np.clip(ridge.predict(x_test), 0, None)
    ridge_metrics = {
        "mae": float(mean_absolute_error(test_reference, ridge_preds)),
        "rmse": float(np.sqrt(mean_squared_error(test_reference, ridge_preds))),
        "nasa_score": nasa_score(ridge_preds, test_reference),
        "r2": float(r2_score(test_reference, ridge_preds)),
    }

    return {
        "Naive Median RUL": median_metrics,
        "Naive Mean RUL": mean_metrics,
        "Ridge Regressor": ridge_metrics,
    }


def evaluate(
    subset: str = DEFAULT_SUBSET,
    artifacts_dir: Path = ARTIFACTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    compare_baselines: bool = True,
    save_reports: bool = True,
) -> dict[str, Any]:
    """Run comprehensive evaluation on the official test set for a given subset."""
    subset = validate_subset(subset)
    train_frame, test_frame, test_rul = load_subset(subset)
    test_reference = test_rul.to_numpy(dtype=np.float64)

    model, scaler, config = load_model_bundle(subset, artifacts_dir=artifacts_dir)
    features = config["features"]
    seq_len = config["sequence_length"]

    test_sequences, engine_ids, last_cycles, _ = latest_windows(
        test_frame, scaler, features, seq_len
    )
    test_preds = predict_rul(test_sequences, model, clip_min=0.0)

    # Calculate primary LSTM metrics
    errors = test_preds - test_reference
    abs_errors = np.abs(errors)
    lstm_mae = float(mean_absolute_error(test_reference, test_preds))
    lstm_rmse = float(np.sqrt(mean_squared_error(test_reference, test_preds)))
    lstm_nasa = float(nasa_score(test_preds, test_reference))
    lstm_r2 = float(r2_score(test_reference, test_preds))

    metrics_summary = {
        "subset": subset,
        "model_type": "LSTM",
        "num_test_engines": len(test_reference),
        "sequence_length": seq_len,
        "feature_count": len(features),
        "metrics": {
            "mae": lstm_mae,
            "rmse": lstm_rmse,
            "nasa_score": lstm_nasa,
            "r2_score": lstm_r2,
            "max_abs_error": float(np.max(abs_errors)),
            "median_abs_error": float(np.median(abs_errors)),
            "percentile_90_abs_error": float(np.percentile(abs_errors, 90)),
        },
    }

    # Baseline comparison
    comparison_records = [
        {
            "Model": "LSTM Regressor (Deep Learning)",
            "MAE (cycles)": round(lstm_mae, 2),
            "RMSE (cycles)": round(lstm_rmse, 2),
            "NASA PHM Score": round(lstm_nasa, 1),
            "R^2 Score": round(lstm_r2, 4),
        }
    ]

    if compare_baselines:
        baseline_results = evaluate_baselines(
            train_frame,
            test_frame,
            test_reference,
            features,
            scaler,
            rul_cap=config.get("rul_cap", 125),
        )
        metrics_summary["baselines"] = baseline_results
        for name, bm in baseline_results.items():
            comparison_records.append(
                {
                    "Model": name,
                    "MAE (cycles)": round(bm["mae"], 2),
                    "RMSE (cycles)": round(bm["rmse"], 2),
                    "NASA PHM Score": round(bm["nasa_score"], 1),
                    "R^2 Score": round(bm["r2"], 4),
                }
            )

    comparison_df = pd.DataFrame(comparison_records)

    # Print evaluation table
    print("\n" + "=" * 70, flush=True)
    print(f"  NASA C-MAPSS {subset} OFFICIAL TEST SET EVALUATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Test engines: {len(test_reference)} | Sequence window: {seq_len} cycles | Active features: {len(features)}")
    print("-" * 70, flush=True)
    print(comparison_df.to_string(index=False), flush=True)
    print("-" * 70, flush=True)
    print(f"Error Distribution: Median={metrics_summary['metrics']['median_abs_error']:.2f} cycles, "
          f"P90={metrics_summary['metrics']['percentile_90_abs_error']:.2f} cycles, "
          f"Max={metrics_summary['metrics']['max_abs_error']:.2f} cycles", flush=True)
    print("=" * 70 + "\n", flush=True)

    if save_reports:
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = reports_dir / f"evaluation_report_{subset.lower()}.json"
        report_csv_path = reports_dir / f"model_comparison_{subset.lower()}.csv"

        report_json_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
        comparison_df.to_csv(report_csv_path, index=False)
        print(f"Saved evaluation report: {report_json_path}", flush=True)
        print(f"Saved model comparison table: {report_csv_path}", flush=True)

    return metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained RUL model against official C-MAPSS test set.")
    parser.add_argument("--subset", choices=("FD001", "FD002", "FD003", "FD004"), default=DEFAULT_SUBSET)
    parser.add_argument("--no-baselines", action="store_true", help="Skip baseline comparison.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(subset=args.subset, compare_baselines=not args.no_baselines)
