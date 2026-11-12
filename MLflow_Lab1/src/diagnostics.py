"""Post-selection diagnostics on held-out data; never used for tuning."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def feature_importance(model, X, y, n_repeats=30):
    """Positive importance means shuffling increases RMSE (target units)."""
    result = permutation_importance(
        model, X, y, scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats, random_state=42, n_jobs=1,
    )
    return pd.DataFrame({
        "feature": X.columns,
        "mean_rmse_increase": result.importances_mean,
        "std_rmse_increase": result.importances_std,
        "n_repeats": n_repeats,
    }).sort_values("mean_rmse_increase", ascending=False).reset_index(drop=True)


def prediction_errors(model, X, y):
    predictions = model.predict(X)
    actual = np.asarray(y)
    return pd.DataFrame({
        "sample_index": X.index,
        "actual": actual,
        "predicted": predictions,
        "residual": actual - predictions,
        "absolute_error": np.abs(actual - predictions),
    })


def save_diagnostics(model, X, y, output, model_name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    importance = feature_importance(model, X, y)
    errors = prediction_errors(model, X, y)
    importance.to_csv(output / "permutation_importance.csv", index=False)
    errors.to_csv(output / "prediction_errors.csv", index=False)

    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ordered.feature, ordered.mean_rmse_increase,
            xerr=ordered.std_rmse_increase, color="#2563eb", capsize=3)
    ax.axvline(0, color="#475569", linewidth=1)
    ax.set(xlabel="Increase in test RMSE after shuffling (mean ± SD, 30 repeats)",
           ylabel="Feature", title=f"Permutation importance • {model_name}")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output / "permutation_importance.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    bounds = [min(errors.actual.min(), errors.predicted.min()),
              max(errors.actual.max(), errors.predicted.max())]
    axes[0].scatter(errors.actual, errors.predicted, alpha=0.7, color="#2563eb")
    axes[0].plot(bounds, bounds, "--", color="#475569", label="Perfect prediction")
    axes[0].set(xlabel="Actual target", ylabel="Predicted target", title="Predicted versus actual")
    axes[0].legend()
    axes[1].scatter(errors.predicted, errors.residual, alpha=0.7, color="#2563eb")
    axes[1].axhline(0, linestyle="--", color="#475569")
    axes[1].set(xlabel="Predicted target", ylabel="Residual (actual − predicted)", title="Residuals versus predictions")
    axes[2].hist(errors.residual, bins=15, color="#2563eb", edgecolor="white")
    axes[2].axvline(0, linestyle="--", color="#475569")
    axes[2].set(xlabel="Residual (actual − predicted)", ylabel="Test observations", title="Residual distribution")
    fig.suptitle(f"Held-out error analysis • {model_name} • n={len(errors)}")
    fig.tight_layout()
    fig.savefig(output / "error_analysis.png", dpi=160)
    plt.close(fig)
    return importance, errors
