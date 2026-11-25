from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import prepare_dataset, validate_dataset, write_raw_dataset


def test_write_and_validate_dataset(sample_frame, tmp_path):
    metadata = write_raw_dataset(sample_frame, "test_run", tmp_path, source="fixture")
    validated = validate_dataset(metadata)

    assert validated["validation_status"] == "passed"
    assert Path(validated["raw_data_path"]).exists()
    report = json.loads(Path(validated["validation_report_path"]).read_text())
    assert report["status"] == "passed"
    assert report["rows"] == len(sample_frame)
    assert report["duplicate_rows"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns=["MedInc"]), "missing required columns"),
        (lambda frame: frame.assign(Latitude=99.0), "Latitude has"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate rows"),
        (lambda frame: frame.assign(MedHouseVal=float("nan")), "null values"),
    ],
)
def test_validation_rejects_bad_data(sample_frame, tmp_path, mutation, message):
    metadata = write_raw_dataset(mutation(sample_frame), "bad_run", tmp_path, source="fixture")
    with pytest.raises(ValueError, match=message):
        validate_dataset(metadata)


def test_prepare_is_deterministic_and_disjoint(sample_frame, tmp_path):
    first = validate_dataset(write_raw_dataset(sample_frame, "first", tmp_path, source="fixture"))
    second = validate_dataset(write_raw_dataset(sample_frame, "second", tmp_path, source="fixture"))
    first = prepare_dataset(first)
    second = prepare_dataset(second)

    first_train = pd.read_csv(first["train_path"])
    first_test = pd.read_csv(first["test_path"])
    second_train = pd.read_csv(second["train_path"])
    second_test = pd.read_csv(second["test_path"])
    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)
    assert len(first_train) + len(first_test) == len(sample_frame)
    assert first_train.merge(first_test, how="inner").empty
