import json
import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from sklearn.base import clone
from src.experiment import (candidates, configure_tracking, evaluate_cv, load_data,
                            metrics, predict_example, select_best, split_data)


def test_dataset_shape_and_raw_features():
    X, y = load_data()
    assert X.shape == (442, 10)
    assert y.shape == (442,)
    assert list(X.columns) == ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    assert X.age.mean() > 20  # raw, not globally standardized
    assert np.isfinite(X.to_numpy()).all()


def test_split_is_reproducible_and_disjoint():
    first, second = split_data(), split_data()
    for a, b in zip(first, second):
        assert a.equals(b)
    assert first[0].shape == (353, 10)
    assert first[1].shape == (89, 10)
    assert set(first[0].index).isdisjoint(first[1].index)


def test_metrics_known_values():
    result = metrics([1, 2, 3], [1, 2, 4])
    assert result == pytest.approx({"mae": 1 / 3, "rmse": np.sqrt(1 / 3), "r2": 0.5})


def test_cv_is_reproducible():
    X, _, y, _ = split_data()
    pipeline = candidates()[1][1]
    assert evaluate_cv(pipeline, X, y) == pytest.approx(evaluate_cv(pipeline, X, y))


def test_selection_ignores_test_performance():
    rows = [{"candidate": "a", "cv_rmse": 10, "test_rmse": 100},
            {"candidate": "b", "cv_rmse": 20, "test_rmse": 1}]
    assert select_best(rows)["candidate"] == "a"


def test_scaler_fits_only_training_data():
    X_train, X_test, y_train, _ = split_data()
    pipeline = candidates()[0][1].fit(X_train, y_train)
    np.testing.assert_allclose(pipeline.steps[0][1].mean_, X_train.mean())
    assert not np.allclose(pipeline.steps[0][1].mean_, load_data()[0].mean())
    assert pipeline.predict(X_test).shape == (89,)


def test_random_forest_reproducibility():
    X, X_test, y, _ = split_data()
    model = candidates()[5][1]
    np.testing.assert_allclose(clone(model).fit(X, y).predict(X_test), clone(model).fit(X, y).predict(X_test))


def test_tracking_and_inference_roundtrip(tmp_path):
    configure_tracking(tmp_path)
    X, X_test, y, _ = split_data()
    model = candidates()[1][1].fit(X, y)
    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics(y, model.predict(X)))
        mlflow.sklearn.log_model(model, "model", pip_requirements=[])
    client = mlflow.MlflowClient()
    assert "rmse" in client.get_run(run.info.run_id).data.metrics
    assert client.list_artifacts(run.info.run_id, "model")
    restored = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
    np.testing.assert_allclose(restored.predict(X_test), model.predict(X_test))
    mlflow.sklearn.save_model(model, str(tmp_path / "model"), pip_requirements=[])
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/best_model.json").write_text(json.dumps({"model_path": "model"}))
    prediction = predict_example(tmp_path)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction, model.predict(X_test.iloc[:1]))
    bad_input = tmp_path / "bad.json"
    bad_input.write_text('[{"age": 42}]')
    with pytest.raises(ValueError, match="features"):
        predict_example(tmp_path, bad_input)


def test_missing_model_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        predict_example(tmp_path)
