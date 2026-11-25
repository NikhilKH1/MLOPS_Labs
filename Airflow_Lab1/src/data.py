from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.common import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    atomic_csv_dump,
    atomic_json_dump,
    run_directory,
    sha256_file,
)

DATASET_SOURCE = "https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html"


def write_raw_dataset(
    frame: pd.DataFrame,
    run_id: str,
    artifact_root: str | Path,
    source: str = DATASET_SOURCE,
) -> dict[str, Any]:
    """Persist a supplied raw frame. Kept separate so tests do not need the network."""
    run_dir = run_directory(artifact_root, run_id)
    raw_path = atomic_csv_dump(frame, run_dir / "raw" / "california_housing.csv")
    metadata = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "raw_data_path": str(raw_path),
        "dataset_source": source,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "raw_sha256": sha256_file(raw_path),
    }
    metadata_path = atomic_json_dump(metadata, run_dir / "raw" / "ingestion_metadata.json")
    metadata["ingestion_metadata_path"] = str(metadata_path)
    return metadata


def ingest_dataset(
    run_id: str,
    artifact_root: str | Path,
    data_home: str | Path | None = None,
) -> dict[str, Any]:
    """Download/cache the public California Housing dataset and persist a raw snapshot."""
    dataset = fetch_california_housing(
        data_home=None if data_home is None else str(data_home),
        as_frame=True,
    )
    frame = dataset.frame.loc[:, ALL_COLUMNS].copy()
    return write_raw_dataset(frame, run_id, artifact_root)


def validate_dataset(metadata: dict[str, Any]) -> dict[str, Any]:
    """Enforce the raw data contract and write a machine-readable validation report."""
    raw_path = Path(metadata["raw_data_path"])
    frame = pd.read_csv(raw_path)
    errors: list[str] = []

    missing_columns = sorted(set(ALL_COLUMNS) - set(frame.columns))
    extra_columns = sorted(set(frame.columns) - set(ALL_COLUMNS))
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")
    if extra_columns:
        errors.append(f"unexpected columns: {extra_columns}")

    available = [column for column in ALL_COLUMNS if column in frame.columns]
    non_numeric = [column for column in available if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        errors.append(f"non-numeric columns: {non_numeric}")

    null_counts = frame[available].isna().sum().astype(int).to_dict() if available else {}
    null_total = sum(null_counts.values())
    if null_total:
        errors.append(f"found {null_total} null values")

    if available:
        numeric_values = frame[available].to_numpy(dtype=float, copy=True)
        non_finite = int((~np.isfinite(numeric_values)).sum())
        if non_finite:
            errors.append(f"found {non_finite} non-finite numeric values")
    else:
        non_finite = 0

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        errors.append(f"found {duplicate_rows} duplicate rows")

    range_rules = {
        "MedInc": (0.0, None),
        "HouseAge": (0.0, None),
        "AveRooms": (0.0, None),
        "AveBedrms": (0.0, None),
        "Population": (0.0, None),
        "AveOccup": (0.0, None),
        "Latitude": (32.0, 42.1),
        "Longitude": (-125.0, -114.0),
        TARGET_COLUMN: (0.0, None),
    }
    range_violations: dict[str, int] = {}
    for column, (minimum, maximum) in range_rules.items():
        if column not in frame:
            continue
        invalid = frame[column] < minimum
        if maximum is not None:
            invalid |= frame[column] > maximum
        count = int(invalid.sum())
        range_violations[column] = count
        if count:
            errors.append(f"{column} has {count} values outside [{minimum}, {maximum}]")

    report = {
        "status": "passed" if not errors else "failed",
        "rows": int(len(frame)),
        "required_columns": ALL_COLUMNS,
        "null_counts": null_counts,
        "non_finite_values": non_finite,
        "duplicate_rows": duplicate_rows,
        "range_violations": range_violations,
        "errors": errors,
    }
    report_path = atomic_json_dump(report, Path(metadata["run_dir"]) / "validation_report.json")
    if errors:
        raise ValueError("Data validation failed: " + "; ".join(errors))

    return {**metadata, "validation_report_path": str(report_path), "validation_status": "passed"}


def prepare_dataset(
    metadata: dict[str, Any],
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Create a deterministic held-out split, stratified on target quantile bins."""
    if metadata.get("validation_status") != "passed":
        raise ValueError("dataset must pass validation before preparation")
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5")

    frame = pd.read_csv(metadata["raw_data_path"])
    target_bins = pd.qcut(frame[TARGET_COLUMN], q=10, labels=False, duplicates="drop")
    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=target_bins,
    )
    train_frame = train_frame.sort_index().reset_index(drop=True)
    test_frame = test_frame.sort_index().reset_index(drop=True)

    split_dir = Path(metadata["run_dir"]) / "prepared"
    train_path = atomic_csv_dump(train_frame, split_dir / "train.csv")
    test_path = atomic_csv_dump(test_frame, split_dir / "test.csv")
    manifest = {
        "random_state": random_state,
        "test_size": test_size,
        "stratification": "10 target quantile bins",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_target_mean": float(train_frame[TARGET_COLUMN].mean()),
        "test_target_mean": float(test_frame[TARGET_COLUMN].mean()),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
    }
    manifest_path = atomic_json_dump(manifest, split_dir / "split_manifest.json")
    return {
        **metadata,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "split_manifest_path": str(manifest_path),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
    }
