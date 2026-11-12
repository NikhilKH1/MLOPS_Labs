import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.diagnostics import feature_importance, prediction_errors, save_diagnostics
from src.experiment import candidates, split_data


def test_importance_detects_signal_and_preserves_input():
    rng = np.random.default_rng(7)
    X = pd.DataFrame({'signal': rng.normal(size=100), 'constant': np.ones(100)})
    y = 10 * X.signal
    model = LinearRegression().fit(X, y)
    before = X.copy(deep=True)
    a = feature_importance(model, X, y, n_repeats=5)
    b = feature_importance(model, X, y, n_repeats=5)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(X, before)
    indexed = a.set_index('feature')
    assert indexed.loc['signal', 'mean_rmse_increase'] > 5
    assert indexed.loc['constant', 'mean_rmse_increase'] == 0


def test_residual_sign_and_sample_identity():
    class FixedModel:
        def predict(self, X):
            return np.array([3., 1.])
    X = pd.DataFrame({'x': [0, 1]}, index=[17, 29])
    result = prediction_errors(FixedModel(), X, [1., 4.])
    assert result.sample_index.tolist() == [17, 29]
    assert result.residual.tolist() == [-2., 3.]
    assert result.absolute_error.tolist() == [2., 3.]


def test_diagnostic_exports_match_held_out_predictions(tmp_path):
    X, X_test, y, y_test = split_data()
    model = candidates()[2][1].fit(X, y)
    importance, errors = save_diagnostics(model, X_test, y_test, tmp_path, 'test-ridge')
    assert len(importance) == 10
    assert len(errors) == 89
    np.testing.assert_allclose(errors.predicted, model.predict(X_test))
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'prediction_errors.csv'), errors)
    for name in ('permutation_importance.png', 'error_analysis.png'):
        assert (tmp_path / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
