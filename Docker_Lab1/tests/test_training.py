"""Tests for the training job (main.py), the shared artifact loader and predict.py."""
import json

import pytest

import predict
from artifacts import load_artifacts, load_test_split
from main import (
    CANDIDATES,
    CV_METRICS,
    MODEL_FILE,
    REPORT_FILE,
    SELECTION_METRIC,
    load_config,
    load_data,
    split_data,
)

CONFIG_VARS = ("MODEL_TYPE", "TEST_SIZE", "RANDOM_STATE", "CV_FOLDS", "OUTPUT_DIR")


def test_dataset_shape_and_labels():
    X, y, feature_names = load_data()
    assert X.shape == (569, 30)
    assert len(feature_names) == 30
    # 212 of the 569 tumours are malignant, encoded as 1 after the label flip
    assert int(y.sum()) == 212


def test_split_is_stratified():
    X, y, _ = load_data()
    _, _, y_train, y_test = split_data(X, y, {"test_size": 0.2, "random_state": 42})
    assert len(y_test) == 114
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_config_defaults(monkeypatch):
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    assert load_config() == {
        "model_type": "auto",
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "output_dir": "output",
    }


@pytest.mark.parametrize("name, value", [
    ("MODEL_TYPE", "xgboost"),
    ("TEST_SIZE", "1.5"),
    ("CV_FOLDS", "1"),
])
def test_config_rejects_invalid_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit, match=name):
        load_config()


def test_training_writes_model_and_report(trained_dir):
    assert (trained_dir / MODEL_FILE).is_file()
    report = json.loads((trained_dir / REPORT_FILE).read_text())

    # Every candidate was cross-validated on every metric, and the best one was kept
    assert set(report["cv_results"]) == set(CANDIDATES)
    for scores in report["cv_results"].values():
        assert set(scores) == set(CV_METRICS)
    best = max(report["cv_results"], key=lambda name: report["cv_results"][name][SELECTION_METRIC]["mean"])
    assert report["model"] == best

    cm = report["confusion_matrix"]["rows_actual_cols_predicted"]
    assert sum(map(sum, cm)) == report["dataset"]["n_test"]


def test_saved_model_matches_reported_accuracy(trained_dir):
    model, report = load_artifacts(trained_dir)
    X_test, y_test, _ = load_test_split(report["config"])
    accuracy = float((model.predict(X_test) == y_test).mean())
    assert accuracy == pytest.approx(report["test_metrics"]["accuracy"], abs=1e-4)
    assert accuracy > 0.9


def test_predict_script(trained_dir, monkeypatch, capsys):
    monkeypatch.setenv("OUTPUT_DIR", str(trained_dir))
    monkeypatch.setattr("sys.argv", ["predict.py", "-n", "5"])
    predict.main()
    out = capsys.readouterr().out
    assert "Loaded" in out
    assert "/5 correct" in out


def test_predict_script_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["predict.py"])
    with pytest.raises(SystemExit, match="No model found"):
        predict.main()
