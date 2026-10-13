from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data import (
    DEFAULT_SUBSET,
    ROOT,
    add_train_targets,
    build_training_examples,
    fit_scaler,
    latest_windows,
    load_subset,
    validate_subset,
)
from src.model import RULLSTM


@dataclass
class TrainingConfig:
    subset: str = DEFAULT_SUBSET
    sequence_length: int = 30
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.15
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 25
    patience: int = 5
    seed: int = 42
    rul_cap: int = 125
    warning_threshold: int = 50
    critical_threshold: int = 20


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _evaluate_loss(
    model: RULLSTM, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    references: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            output = model(features)
            losses.append(float(loss_fn(output, labels).item()))
            predictions.append(output.detach().cpu().numpy())
            references.append(labels.detach().cpu().numpy())
    return (
        float(np.mean(losses)),
        np.concatenate(predictions),
        np.concatenate(references),
    )


def nasa_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    error = predicted - actual
    scores = np.where(error < 0, np.exp(-error / 13.0) - 1, np.exp(error / 10.0) - 1)
    return float(scores.sum())


def _artifact_path(directory, prefix: str, subset: str, suffix: str):
    return directory / f"{prefix}_{subset.lower()}.{suffix}"


def train(config: TrainingConfig) -> dict:
    config.subset = validate_subset(config.subset)
    if min(config.sequence_length, config.hidden_size, config.num_layers, config.batch_size, config.epochs, config.patience, config.rul_cap) < 1:
        raise ValueError("Sequence, model, training, patience, and RUL-cap settings must be positive.")
    if config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
    if not 0 <= config.dropout < 1:
        raise ValueError("Dropout must be in the interval [0, 1).")
    if config.critical_threshold >= config.warning_threshold:
        raise ValueError("The critical threshold must be lower than the warning threshold.")

    _seed_everything(config.seed)
    train_frame, test_frame, test_rul = load_subset(config.subset)
    train_frame = add_train_targets(train_frame, config.rul_cap)
    engine_ids = np.sort(train_frame["engine_id"].unique())
    if len(engine_ids) < 2:
        raise ValueError("At least two training engines are required for a grouped validation split.")
    train_ids, validation_ids = train_test_split(
        engine_ids, test_size=0.2, random_state=config.seed
    )

    scaler, features = fit_scaler(train_frame[train_frame["engine_id"].isin(train_ids)])
    train_x, train_y = build_training_examples(
        train_frame, train_ids, scaler, features, config.sequence_length
    )
    validation_x, validation_y = build_training_examples(
        train_frame, validation_ids, scaler, features, config.sequence_length
    )
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_x), torch.from_numpy(validation_y)
    )
    generator = torch.Generator().manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size * 2, pin_memory=device.type == "cuda"
    )

    model = RULLSTM(
        len(features), config.hidden_size, config.num_layers, config.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.SmoothL1Loss()
    artifact_dir = ROOT / "artifacts"
    report_dir = ROOT / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _artifact_path(artifact_dir, "model", config.subset, "pt")
    scaler_path = _artifact_path(artifact_dir, "scaler", config.subset, "joblib")
    config_path = _artifact_path(artifact_dir, "config", config.subset, "json")
    history_path = _artifact_path(report_dir, "training_history", config.subset, "csv")
    metrics_path = _artifact_path(artifact_dir, "metrics", config.subset, "json")
    predictions_path = _artifact_path(report_dir, "test_predictions", config.subset, "csv")

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    print(
        f"Training {config.subset} LSTM on {device}; "
        f"{len(train_ids)} training engines, {len(validation_ids)} validation engines, "
        f"{len(features)} features.",
        flush=True,
    )
    for epoch in range(1, config.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features_batch), labels_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        validation_loss, validation_prediction, validation_reference = _evaluate_loss(
            model, validation_loader, loss_fn, device
        )
        validation_prediction = np.clip(validation_prediction, 0, config.rul_cap)
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "validation_loss": validation_loss,
            "validation_mae": float(mean_absolute_error(validation_reference, validation_prediction)),
            "validation_rmse": float(np.sqrt(mean_squared_error(validation_reference, validation_prediction))),
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:02d}/{config.epochs} "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_MAE={epoch_record['validation_mae']:.2f}",
            flush=True,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_size": len(features),
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "dropout": config.dropout,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.", flush=True)
                break

    if best_epoch == 0:
        raise RuntimeError("Training did not produce a valid validation checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    validation_loss, validation_prediction, validation_reference = _evaluate_loss(
        model, validation_loader, loss_fn, device
    )
    validation_prediction = np.clip(validation_prediction, 0, config.rul_cap)
    test_x, test_engine_ids, last_cycles, _ = latest_windows(
        test_frame, scaler, features, config.sequence_length
    )
    if test_engine_ids != sorted(test_engine_ids):
        raise RuntimeError("Test engine sequences are not in the official ascending engine order.")
    test_dataset = TensorDataset(torch.from_numpy(test_x))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2)
    test_batches: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in test_loader:
            test_batches.append(model(batch.to(device)).detach().cpu().numpy())
    test_prediction = np.clip(np.concatenate(test_batches), 0, None)
    test_reference = test_rul.to_numpy(dtype=np.float64)
    if len(test_prediction) != len(test_reference):
        raise RuntimeError("Test predictions and official RUL labels have different lengths.")

    prediction_report = pd.DataFrame(
        {
            "engine_id": test_engine_ids,
            "last_observed_cycle": last_cycles,
            "actual_rul_cycles": test_reference,
            "predicted_rul_cycles": test_prediction,
        }
    )
    prediction_report.to_csv(predictions_path, index=False)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(history_path, index=False)
    joblib.dump(scaler, scaler_path)
    saved_config = {
        **asdict(config),
        "features": features,
        "model_type": "LSTM",
        "trained_engine_ids": [int(engine_id) for engine_id in train_ids],
        "validation_engine_ids": [int(engine_id) for engine_id in validation_ids],
    }
    config_path.write_text(json.dumps(saved_config, indent=2), encoding="utf-8")

    results = {
        "subset": config.subset,
        "model_type": "LSTM",
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "validation_engines": int(len(validation_ids)),
        "validation_mae": float(mean_absolute_error(validation_reference, validation_prediction)),
        "validation_rmse": float(np.sqrt(mean_squared_error(validation_reference, validation_prediction))),
        "test_engines": int(len(test_engine_ids)),
        "test_mae": float(mean_absolute_error(test_reference, test_prediction)),
        "test_rmse": float(np.sqrt(mean_squared_error(test_reference, test_prediction))),
        "test_nasa_score": nasa_score(test_prediction, test_reference),
        "test_r2_score": float(r2_score(test_reference, test_prediction)),
        "sequence_length": config.sequence_length,
        "feature_count": len(features),
        "rul_cap": config.rul_cap,
        "prediction_report": str(predictions_path.relative_to(ROOT)),
        "training_history": str(history_path.relative_to(ROOT)),
        "test_predictions": prediction_report.to_dict(orient="records"),
    }
    metrics_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved model, scaler, config, metrics, and reports for {config.subset}.", flush=True)
    print(json.dumps({key: value for key, value in results.items() if key != "test_predictions"}, indent=2), flush=True)
    return results


def parse_args() -> TrainingConfig:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train and evaluate a C-MAPSS LSTM RUL model.")
    parser.add_argument("--subset", choices=("FD001", "FD002", "FD003", "FD004"), default=defaults.subset)
    parser.add_argument("--sequence-length", type=int, default=defaults.sequence_length)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--rul-cap", type=int, default=defaults.rul_cap)
    parser.add_argument("--warning-threshold", type=int, default=defaults.warning_threshold)
    parser.add_argument("--critical-threshold", type=int, default=defaults.critical_threshold)
    return TrainingConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
