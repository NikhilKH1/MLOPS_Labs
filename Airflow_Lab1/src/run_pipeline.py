from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import ingest_dataset, prepare_dataset, validate_dataset
from src.evaluation import evaluate_model
from src.reporting import publish_report
from src.training import train_and_select


def run_pipeline(
    run_id: str,
    artifact_root: str | Path,
    data_home: str | Path | None = None,
    fast: bool = False,
) -> dict:
    metadata = ingest_dataset(run_id, artifact_root, data_home=data_home)
    metadata = validate_dataset(metadata)
    metadata = prepare_dataset(metadata)
    metadata = train_and_select(metadata, cv_folds=3 if fast else 5, fast=fast)
    metadata = evaluate_model(metadata, fast=fast)
    return publish_report(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the California Housing ML pipeline locally.")
    parser.add_argument("--run-id", default="local_manual")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--data-home", default=".cache/scikit-learn")
    parser.add_argument("--fast", action="store_true", help="Use smaller models for a smoke test.")
    args = parser.parse_args()
    result = run_pipeline(
        run_id=args.run_id,
        artifact_root=args.artifact_root,
        data_home=args.data_home,
        fast=args.fast,
    )
    summary = {
        "run_id": result["run_id"],
        "selected_model": result["selected_model"],
        "test_rmse": result["test_rmse"],
        "test_r2": result["test_r2"],
        "model_card_path": result["model_card_path"],
        "run_manifest_path": result["run_manifest_path"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
