from __future__ import annotations

from pathlib import Path

import pytest

airflow = pytest.importorskip("airflow")
from airflow.models import DagBag


def test_dag_imports_and_has_expected_graph():
    dags_folder = Path(__file__).resolve().parents[1] / "dags"
    dag_bag = DagBag(dag_folder=str(dags_folder), include_examples=False)
    assert dag_bag.import_errors == {}

    dag = dag_bag.dags.get("california_housing_ml_pipeline")
    assert dag is not None
    assert set(dag.task_ids) == {"ingest", "validate", "prepare", "train", "evaluate", "publish"}
    expected = {
        "ingest": {"validate"},
        "validate": {"prepare"},
        "prepare": {"train"},
        "train": {"evaluate"},
        "evaluate": {"publish"},
        "publish": set(),
    }
    for task_id, downstream in expected.items():
        assert dag.get_task(task_id).downstream_task_ids == downstream
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    assert dag.get_task("ingest").retries == 2
