from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    atomic_csv_dump,
    atomic_json_dump,
)


def _save_diagnostics(predictions: pd.DataFrame, path: Path) -> None:
    actual = predictions["actual"].to_numpy()
    predicted = predictions["predicted"].to_numpy()
    residual = predictions["residual"].to_numpy()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(actual, predicted, alpha=0.35, s=14, color="#2563eb")
    limits = [float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))]
    axes[0].plot(limits, limits, linestyle="--", color="#111827")
    axes[0].set(title="Predicted vs actual", xlabel="Actual value ($100,000s)", ylabel="Prediction")

    axes[1].scatter(predicted, residual, alpha=0.35, s=14, color="#dc2626")
    axes[1].axhline(0, linestyle="--", color="#111827")
    axes[1].set(title="Residuals", xlabel="Prediction", ylabel="Actual - predicted")

    axes[2].hist(residual, bins=35, color="#0f766e", edgecolor="white")
    axes[2].axvline(0, linestyle="--", color="#111827")
    axes[2].set(title="Residual distribution", xlabel="Residual", ylabel="Count")

    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _regional_slices(test_frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    labels = ["southmost", "south_central", "north_central", "northmost"]
    regions, boundaries = pd.qcut(
        test_frame["Latitude"],
        q=4,
        labels=labels,
        retbins=True,
        duplicates="drop",
    )

    rows: list[dict[str, Any]] = []
    for index, region in enumerate(labels):
        mask = regions == region
        actual = predictions.loc[mask, "actual"]
        predicted = predictions.loc[mask, "predicted"]
        rows.append(
            {
                "region": region,
                "definition": f"latitude in [{boundaries[index]:.3f}, {boundaries[index + 1]:.3f}]",
                "count": int(mask.sum()),
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
                "r2": float(r2_score(actual, predicted)),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def evaluate_model(
    metadata: dict[str, Any],
    permutation_repeats: int = 10,
    fast: bool = False,
) -> dict[str, Any]:
    """Evaluate the selected pipeline once on held-out data and generate diagnostics."""
    test_frame = pd.read_csv(metadata["test_path"])
    X_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame[TARGET_COLUMN]
    model = joblib.load(metadata["model_path"])
    predicted = model.predict(X_test)
    residual = y_test.to_numpy() - predicted

    metrics = {
        "selected_model": metadata["selected_model"],
        "test_rows": int(len(test_frame)),
        "mae": float(mean_absolute_error(y_test, predicted)),
        "rmse": float(mean_squared_error(y_test, predicted) ** 0.5),
        "r2": float(r2_score(y_test, predicted)),
        "mean_residual": float(np.mean(residual)),
        "median_absolute_error": float(np.median(np.abs(residual))),
    }
    predictions = test_frame.loc[:, ["Latitude", "Longitude", TARGET_COLUMN]].copy()
    predictions = predictions.rename(columns={TARGET_COLUMN: "actual"})
    predictions["predicted"] = predicted
    predictions["residual"] = residual
    predictions["absolute_error"] = np.abs(residual)

    evaluation_dir = Path(metadata["run_dir"]) / "evaluation"
    predictions_path = atomic_csv_dump(predictions, evaluation_dir / "predictions.csv")
    metrics_path = atomic_json_dump(metrics, evaluation_dir / "metrics.json")

    diagnostics_path = evaluation_dir / "residual_diagnostics.png"
    _save_diagnostics(predictions, diagnostics_path)

    repeats = 3 if fast else permutation_repeats
    importance_result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        n_jobs=1 if fast else -1,
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "rmse_increase_mean": importance_result.importances_mean,
            "rmse_increase_std": importance_result.importances_std,
            "repeats": repeats,
        }
    ).sort_values("rmse_increase_mean", ascending=False)
    importance_path = atomic_csv_dump(importance, evaluation_dir / "permutation_importance.csv")

    figure, axis = plt.subplots(figsize=(8, 5))
    ordered = importance.sort_values("rmse_increase_mean")
    axis.barh(
        ordered["feature"],
        ordered["rmse_increase_mean"],
        xerr=ordered["rmse_increase_std"],
        color="#2563eb",
        alpha=0.85,
    )
    axis.axvline(0, color="#111827", linewidth=1)
    axis.set(title="Held-out permutation importance", xlabel="Increase in RMSE after shuffling")
    figure.tight_layout()
    importance_plot_path = evaluation_dir / "permutation_importance.png"
    figure.savefig(importance_plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    slices = _regional_slices(test_frame.reset_index(drop=True), predictions.reset_index(drop=True))
    slices_path = atomic_csv_dump(slices, evaluation_dir / "regional_slices.csv")

    return {
        **metadata,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "diagnostics_path": str(diagnostics_path),
        "importance_path": str(importance_path),
        "importance_plot_path": str(importance_plot_path),
        "regional_slices_path": str(slices_path),
        "test_rmse": metrics["rmse"],
        "test_r2": metrics["r2"],
    }
