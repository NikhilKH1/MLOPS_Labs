from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow.sdk import dag, get_current_context, task

from src.data import ingest_dataset, prepare_dataset, validate_dataset
from src.evaluation import evaluate_model
from src.reporting import publish_report
from src.training import train_and_select

ARTIFACT_ROOT = os.getenv("PIPELINE_ARTIFACT_ROOT", str(PROJECT_ROOT / "artifacts"))
DATA_HOME = os.getenv("SKLEARN_DATA", str(PROJECT_ROOT / ".cache" / "scikit-learn"))


@dag(
    dag_id="california_housing_ml_pipeline",
    description="Validate data, select a regression model, evaluate it, and publish a model card.",
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "Nikhil Kundalli Harish",
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
    },
    tags=["assignment-5", "ml", "california-housing"],
)
def california_housing_ml_pipeline():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def ingest() -> dict:
        context = get_current_context()
        return ingest_dataset(context["run_id"], ARTIFACT_ROOT, data_home=DATA_HOME)

    @task
    def validate(metadata: dict) -> dict:
        return validate_dataset(metadata)

    @task
    def prepare(metadata: dict) -> dict:
        return prepare_dataset(metadata)

    @task
    def train(metadata: dict) -> dict:
        return train_and_select(metadata)

    @task
    def evaluate(metadata: dict) -> dict:
        return evaluate_model(metadata)

    @task
    def publish(metadata: dict) -> dict:
        result = publish_report(metadata)
        return {
            "run_id": result["run_id"],
            "selected_model": result["selected_model"],
            "test_rmse": result["test_rmse"],
            "test_r2": result["test_r2"],
            "model_card_path": result["model_card_path"],
            "run_manifest_path": result["run_manifest_path"],
        }

    publish(evaluate(train(prepare(validate(ingest())))))


california_housing_ml_pipeline()
