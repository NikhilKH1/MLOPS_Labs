from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_COLUMN = "MedHouseVal"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
RANDOM_STATE = 42


def safe_run_id(run_id: str) -> str:
    """Return a filesystem-safe, deterministic representation of an Airflow run ID."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("._")
    if not cleaned:
        raise ValueError("run_id must contain at least one filesystem-safe character")
    return cleaned[:160]


def run_directory(artifact_root: str | Path, run_id: str) -> Path:
    path = Path(artifact_root).expanduser().resolve() / safe_run_id(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json_dump(payload: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def atomic_csv_dump(frame: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    """Convert NumPy scalar-like values into ordinary JSON values."""
    if hasattr(value, "item"):
        return value.item()
    return value
