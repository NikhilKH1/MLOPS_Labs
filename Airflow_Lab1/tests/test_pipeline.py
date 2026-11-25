from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import prepare_dataset, validate_dataset, write_raw_dataset
from src.evaluation import evaluate_model
from src.reporting import publish_report
from src.training import train_and_select


def test_end_to_end_functions_generate_submission_artifacts(sample_frame, tmp_path):
    metadata = write_raw_dataset(sample_frame, "integration_run", tmp_path, source="fixture")
    metadata = validate_dataset(metadata)
    metadata = prepare_dataset(metadata)
    metadata = train_and_select(metadata, cv_folds=3, fast=True)
    metadata = evaluate_model(metadata, fast=True)
    result = publish_report(metadata)

    metrics = json.loads(Path(result["metrics_path"]).read_text())
    predictions = pd.read_csv(result["predictions_path"])
    manifest = json.loads(Path(result["run_manifest_path"]).read_text())

    assert metrics["rmse"] >= 0
    assert metrics["r2"] > 0.5
    assert len(predictions) == result["test_rows"]
    assert Path(result["diagnostics_path"]).stat().st_size > 10_000
    assert Path(result["importance_plot_path"]).stat().st_size > 10_000
    assert "educational model" in Path(result["model_card_path"]).read_text()
    assert len(manifest["artifacts"]) == result["artifact_count"]


def test_manifest_contains_checksums(sample_frame, tmp_path):
    metadata = write_raw_dataset(sample_frame, "manifest_run", tmp_path, source="fixture")
    metadata = train_and_select(prepare_dataset(validate_dataset(metadata)), cv_folds=2, fast=True)
    result = publish_report(evaluate_model(metadata, fast=True))
    manifest = json.loads(Path(result["run_manifest_path"]).read_text())
    assert manifest["artifacts"]
    assert all(len(item["sha256"]) == 64 for item in manifest["artifacts"])
    assert all(item["path"] != "run_manifest.json" for item in manifest["artifacts"])

    repeated = publish_report(result)
    repeated_manifest = json.loads(Path(repeated["run_manifest_path"]).read_text())
    assert len(repeated_manifest["artifacts"]) == repeated["artifact_count"]
