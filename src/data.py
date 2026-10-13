from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_SUBSET = "FD003"
COLS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + [
    f"s{i}" for i in range(1, 22)
]
FEATURE_CANDIDATES = COLS[2:]


def validate_subset(subset: str) -> str:
    normalized = subset.upper()
    if normalized not in SUBSETS:
        raise ValueError(f"Unsupported subset {subset!r}; choose one of {', '.join(SUBSETS)}.")
    return normalized


def load_subset(subset: str = DEFAULT_SUBSET) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    subset = validate_subset(subset)
    paths = {
        "train": DATA_DIR / f"train_{subset}.txt",
        "test": DATA_DIR / f"test_{subset}.txt",
        "rul": DATA_DIR / f"RUL_{subset}.txt",
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required dataset file(s) in {DATA_DIR}: {', '.join(missing)}")

    train = _read_trajectories(paths["train"], subset, "training")
    test = _read_trajectories(paths["test"], subset, "test")
    rul_frame = pd.read_csv(paths["rul"], sep=r"\s+", header=None, dtype=np.float64)
    if rul_frame.shape[1] != 1 or rul_frame.empty:
        raise ValueError(f"{paths['rul'].name} must contain exactly one RUL value per line.")
    rul = rul_frame.iloc[:, 0].rename("rul")
    if not np.isfinite(rul.to_numpy()).all() or (rul < 0).any():
        raise ValueError(f"{paths['rul'].name} contains invalid or negative RUL values.")
    test_engine_ids = sorted(int(value) for value in test["engine_id"].unique())
    if len(rul) != len(test_engine_ids):
        raise ValueError(
            f"{paths['rul'].name} contains {len(rul)} labels for "
            f"{len(test_engine_ids)} test engines."
        )
    return train, test, rul.reset_index(drop=True)


def _read_trajectories(path: Path, subset: str, split_name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLS, dtype=np.float64)
    except (pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"{path.name} is not a valid {subset} C-MAPSS {split_name} file: {exc}") from exc
    if frame.empty or frame.shape[1] != len(COLS):
        raise ValueError(f"{path.name} must contain 26 whitespace-separated numeric columns.")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path.name} contains missing or non-finite values.")
    if (frame["engine_id"] < 1).any() or (frame["cycle"] < 1).any():
        raise ValueError(f"{path.name} contains invalid engine IDs or cycle numbers.")
    if frame.duplicated(["engine_id", "cycle"]).any():
        raise ValueError(f"{path.name} contains duplicate engine/cycle observations.")
    return frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def add_train_targets(frame: pd.DataFrame, rul_cap: int | None = None) -> pd.DataFrame:
    result = frame.copy()
    max_cycle = result.groupby("engine_id")["cycle"].transform("max")
    result["rul"] = max_cycle - result["cycle"]
    if rul_cap is not None:
        if rul_cap < 1:
            raise ValueError("The RUL clipping threshold must be positive.")
        result["rul"] = result["rul"].clip(upper=rul_cap)
    return result


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select non-constant operating settings and sensors from training rows."""
    return [
        column
        for column in FEATURE_CANDIDATES
        if column in frame.columns and frame[column].nunique(dropna=False) > 1
    ]


def make_window(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size < 1:
        raise ValueError("The sequence length must be positive.")
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("A window requires a non-empty two-dimensional feature array.")
    recent = values[-window_size:]
    if len(recent) < window_size:
        padding = np.repeat(recent[:1], window_size - len(recent), axis=0)
        recent = np.concatenate((padding, recent), axis=0)
    return recent.astype(np.float32, copy=False)


def latest_windows(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str],
    window_size: int,
) -> tuple[np.ndarray, list[int], list[int], dict[int, pd.DataFrame]]:
    windows: list[np.ndarray] = []
    engine_ids: list[int] = []
    last_cycles: list[int] = []
    histories: dict[int, pd.DataFrame] = {}
    for engine_id, group in frame.groupby("engine_id", sort=True):
        group = group.sort_values("cycle")
        values = scaler.transform(group[feature_cols].to_numpy(dtype=np.float32))
        windows.append(make_window(values, window_size))
        engine_ids.append(int(engine_id))
        last_cycles.append(int(group["cycle"].iloc[-1]))
        histories[int(engine_id)] = group
    if not windows:
        raise ValueError("No engine trajectories were found.")
    return np.stack(windows), engine_ids, last_cycles, histories


def fit_scaler(
    training_rows: pd.DataFrame,
) -> tuple[StandardScaler, list[str]]:
    features = feature_columns(training_rows)
    if not features:
        raise ValueError("No non-constant operating settings or sensors are available.")
    scaler = StandardScaler()
    scaler.fit(training_rows[features].to_numpy(dtype=np.float32))
    return scaler, features


def build_training_examples(
    frame: pd.DataFrame,
    engine_ids: np.ndarray,
    scaler: StandardScaler,
    feature_cols: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    targets: list[float] = []
    selected = frame[frame["engine_id"].isin(engine_ids)]
    for _, group in selected.groupby("engine_id", sort=True):
        group = group.sort_values("cycle")
        values = scaler.transform(group[feature_cols].to_numpy(dtype=np.float32))
        labels = group["rul"].to_numpy(dtype=np.float32)
        for index, target in enumerate(labels):
            start = max(0, index - window_size + 1)
            windows.append(make_window(values[start : index + 1], window_size))
            targets.append(float(target))
    if not windows:
        raise ValueError("The selected engine IDs produced no training examples.")
    return np.stack(windows), np.asarray(targets, dtype=np.float32)


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.replace({np.nan: None}).to_dict(orient="records")
