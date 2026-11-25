from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import atomic_json_dump, sha256_file


def _markdown_table(frame: pd.DataFrame, float_columns: set[str]) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---:" if column in float_columns else "---" for column in columns) + "|"
    rows = []
    for record in frame.to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            values.append(f"{value:.4f}" if column in float_columns else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def publish_report(metadata: dict[str, Any]) -> dict[str, Any]:
    """Create a human-readable model card and a checksum manifest for the run."""
    run_dir = Path(metadata["run_dir"])
    metrics = json.loads(Path(metadata["metrics_path"]).read_text(encoding="utf-8"))
    training = json.loads(Path(metadata["training_report_path"]).read_text(encoding="utf-8"))
    validation = json.loads(Path(metadata["validation_report_path"]).read_text(encoding="utf-8"))
    comparison = pd.read_csv(metadata["model_comparison_path"])
    slices = pd.read_csv(metadata["regional_slices_path"])

    comparison_table = _markdown_table(
        comparison[["model", "cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_r2_mean"]],
        {"cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_r2_mean"},
    )
    slice_table = _markdown_table(
        slices[["region", "count", "mae", "rmse", "r2"]],
        {"mae", "rmse", "r2"},
    )
    created_at = datetime.now(timezone.utc).isoformat()
    model_card = f"""# California Housing Model Card

## Overview

- Run ID: `{metadata['run_id']}`
- Generated (UTC): `{created_at}`
- Dataset rows: {metadata['rows']:,}
- Train/test rows: {metadata['train_rows']:,} / {metadata['test_rows']:,}
- Selected model: **{training['selected_model']}**
- Selection rule: {training['selection_metric']}
- Data validation: **{validation['status']}**

The target is median home value for a California census-block group, expressed
in units of $100,000. This educational model is not suitable for appraisal,
lending, insurance, or other high-impact decisions.

## Model selection

Candidates were compared using training-only cross-validation. The held-out test
set was not used to select the winner.

{comparison_table}

## Held-out results

| Metric | Value |
|---|---:|
| MAE | {metrics['mae']:.4f} |
| RMSE | {metrics['rmse']:.4f} |
| R-squared | {metrics['r2']:.4f} |
| Mean residual | {metrics['mean_residual']:.4f} |

## Geographic error slices

These four diagnostic bands are defined by held-out latitude quartiles, so each
contains approximately the same number of rows. They are not demographic
fairness groups.

{slice_table}

## Artifacts

- `training/model.joblib`: complete fitted preprocessing/model pipeline
- `training/model_comparison.csv`: cross-validation results
- `evaluation/metrics.json`: held-out metrics
- `evaluation/predictions.csv`: held-out predictions and residuals
- `evaluation/residual_diagnostics.png`: prediction/residual plots
- `evaluation/permutation_importance.csv` and `.png`: held-out feature importance
- `evaluation/regional_slices.csv`: geographic diagnostic slices

## Limitations

- The data comes from the 1990 U.S. census and does not represent today's market.
- Census-block averages hide variation between individual homes and residents.
- A single deterministic split does not measure every source of uncertainty.
- Geographic features can proxy for socioeconomic and demographic differences.
- Permutation importance describes model reliance, not causal effects.
"""
    model_card_path = run_dir / "MODEL_CARD.md"
    model_card_path.write_text(model_card, encoding="utf-8")

    artifact_paths = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    )
    manifest = {
        "run_id": metadata["run_id"],
        "created_at_utc": created_at,
        "selected_model": metadata["selected_model"],
        "test_rmse": metadata["test_rmse"],
        "test_r2": metadata["test_r2"],
        "artifacts": [
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    manifest_path = atomic_json_dump(manifest, run_dir / "run_manifest.json")
    return {
        **metadata,
        "model_card_path": str(model_card_path),
        "run_manifest_path": str(manifest_path),
        "artifact_count": len(artifact_paths),
    }
