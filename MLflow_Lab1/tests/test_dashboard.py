"""Exercise the app without requiring a pre-existing local model or server."""
import json
import shutil
from pathlib import Path

import mlflow.sklearn
import pytest
from streamlit.testing.v1 import AppTest

import src.dashboard as dashboard
from src.experiment import candidates, split_data

APP = Path(__file__).resolve().parents[1] / 'dashboard.py'


@pytest.fixture
def dashboard_root(tmp_path, monkeypatch):
    shutil.copytree(APP.parent / 'outputs', tmp_path / 'outputs')
    monkeypatch.setattr(dashboard, 'ROOT', tmp_path)
    return tmp_path


def test_dashboard_views_and_filters(dashboard_root):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 9
    app.multiselect[0].set_value(['Ridge']).run()
    assert len(app.dataframe[0].value) == 4
    app.selectbox[0].select('test_r2').run()
    values = app.dataframe[0].value.test_r2.tolist()
    assert values == sorted(values, reverse=True)
    app.multiselect[0].set_value([]).run()
    assert 'Select at least' in app.info[0].value
    app.sidebar.radio[0].set_value('Feature importance').run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 10
    app.sidebar.radio[0].set_value('Error analysis').run()
    app.slider[0].set_value(100.0).run()
    assert (app.dataframe[0].value.absolute_error >= 100).all()
    assert not app.exception


def test_dashboard_prediction_matches_saved_pipeline(dashboard_root):
    X, X_test, y, _ = split_data()
    model = candidates()[2][1].fit(X, y)
    mlflow.sklearn.save_model(model, str(dashboard_root / 'model'), pip_requirements=[])
    summary_path = dashboard_root / 'outputs/best_model.json'
    summary = json.loads(summary_path.read_text())
    summary['model_path'] = 'model'
    summary_path.write_text(json.dumps(summary))
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.radio[0].set_value('Prediction').run()
    app.button[0].click().run()  # Main-area form submit; sidebar refresh is last.
    assert not app.exception
    prediction = next(m.value for m in app.metric if m.label == 'Predicted progression')
    assert float(prediction) == pytest.approx(float(model.predict(X_test.iloc[:1])[0]), abs=0.001)
    app.number_input(key='Held-out example_sex').set_value(3.0)
    app.button[0].click().run()
    assert 'sex code' in app.error[0].value


def test_dashboard_missing_model_and_missing_results(dashboard_root):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.radio[0].set_value('Prediction').run()
    assert not app.exception
    assert 'locally trained model' in app.info[0].value
    (dashboard_root / 'outputs/best_model.json').unlink()
    app.run()
    assert not app.exception
    assert 'Results are unavailable' in app.info[0].value
