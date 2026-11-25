from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.common import FEATURE_COLUMNS, TARGET_COLUMN
from src.data import prepare_dataset, validate_dataset, write_raw_dataset
from src.training import candidate_models, train_and_select


def _prepared(sample_frame, tmp_path):
    metadata = write_raw_dataset(sample_frame, "training_run", tmp_path, source="fixture")
    return prepare_dataset(validate_dataset(metadata))


def test_candidates_are_complete_pipelines():
    models = candidate_models(fast=True)
    assert set(models) == {"elastic_net", "random_forest", "hist_gradient_boosting"}
    assert all(hasattr(model, "steps") for model in models.values())


def test_training_selects_and_reloads_model(sample_frame, tmp_path):
    trained = train_and_select(_prepared(sample_frame, tmp_path), cv_folds=3, fast=True)
    comparison = pd.read_csv(trained["model_comparison_path"])
    report = json.loads(Path(trained["training_report_path"]).read_text())
    model = joblib.load(trained["model_path"])
    test_frame = pd.read_csv(trained["test_path"])

    assert len(comparison) == 3
    assert comparison.iloc[0]["model"] == trained["selected_model"]
    assert report["selected_model"] == trained["selected_model"]
    predictions = model.predict(test_frame[FEATURE_COLUMNS])
    assert len(predictions) == len(test_frame[TARGET_COLUMN])
