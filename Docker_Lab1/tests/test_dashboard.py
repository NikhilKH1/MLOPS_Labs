"""Run the Streamlit dashboard headlessly with streamlit.testing and check what it shows."""
import pytest
from streamlit.testing.v1 import AppTest

from artifacts import load_artifacts, load_test_split
from main import CLASS_NAMES


@pytest.fixture
def run_dashboard(app_dir, monkeypatch):
    def run(directory):
        monkeypatch.setenv("OUTPUT_DIR", str(directory))
        return AppTest.from_file(str(app_dir / "dashboard.py"), default_timeout=60).run()
    return run


def test_dashboard_shows_metrics_and_sections(trained_dir, run_dashboard):
    at = run_dashboard(trained_dir)
    assert not at.exception

    _, report = load_artifacts(trained_dir)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Accuracy"] == f"{report['test_metrics']['accuracy']:.3f}"
    assert metrics["F1 score"] == f"{report['test_metrics']['f1']:.3f}"
    assert {"Held-out test set", "Model comparison", "Confusion matrix", "Try a prediction"} <= {
        s.value for s in at.subheader
    }


def test_dashboard_prediction_matches_model(trained_dir, run_dashboard):
    at = run_dashboard(trained_dir)
    at.selectbox[0].select(3).run()
    assert not at.exception

    model, report = load_artifacts(trained_dir)
    X_test, _, _ = load_test_split(report["config"])
    expected = CLASS_NAMES[int(model.predict(X_test[3:4])[0])]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Prediction"] == expected.capitalize()


def test_dashboard_misclassified_filter(trained_dir, run_dashboard):
    at = run_dashboard(trained_dir)
    at.checkbox[0].check().run()
    assert not at.exception
    assert all("misclassified" in option for option in at.selectbox[0].options)


def test_dashboard_without_model_shows_error(tmp_path, run_dashboard):
    at = run_dashboard(tmp_path)
    assert not at.exception
    assert "No trained model" in at.error[0].value
