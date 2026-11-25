from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    atomic_csv_dump,
    atomic_json_dump,
    sha256_file,
)


def candidate_models(fast: bool = False) -> dict[str, Pipeline]:
    """Return deterministic candidate pipelines; fast mode is used only by tests."""
    return {
        "elastic_net": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.001,
                        l1_ratio=0.1,
                        max_iter=20_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=30 if fast else 200,
                        max_depth=12 if fast else 18,
                        min_samples_leaf=2,
                        n_jobs=1 if fast else -1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.08,
                        max_iter=50 if fast else 250,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


def train_and_select(
    metadata: dict[str, Any],
    cv_folds: int = 5,
    fast: bool = False,
) -> dict[str, Any]:
    """Cross-validate candidates on training data, refit the winner, and save it."""
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    train_frame = pd.read_csv(metadata["train_path"])
    X_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame[TARGET_COLUMN]
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    rows: list[dict[str, Any]] = []
    models = candidate_models(fast=fast)
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1 if fast else -1,
            error_score="raise",
        )
        rows.append(
            {
                "model": name,
                "cv_rmse_mean": float(-np.mean(scores["test_rmse"])),
                "cv_rmse_std": float(np.std(-scores["test_rmse"], ddof=0)),
                "cv_mae_mean": float(-np.mean(scores["test_mae"])),
                "cv_mae_std": float(np.std(-scores["test_mae"], ddof=0)),
                "cv_r2_mean": float(np.mean(scores["test_r2"])),
                "cv_r2_std": float(np.std(scores["test_r2"], ddof=0)),
                "fit_seconds_mean": float(np.mean(scores["fit_time"])),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    selected_name = str(comparison.iloc[0]["model"])
    selected_model = models[selected_name]
    selected_model.fit(X_train, y_train)

    training_dir = Path(metadata["run_dir"]) / "training"
    comparison_path = atomic_csv_dump(comparison, training_dir / "model_comparison.csv")
    model_path = training_dir / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(".joblib.tmp")
    joblib.dump(selected_model, temporary_model)
    os.replace(temporary_model, model_path)

    report = {
        "selected_model": selected_name,
        "selection_metric": "lowest mean cross-validation RMSE",
        "selected_cv_rmse": float(comparison.iloc[0]["cv_rmse_mean"]),
        "cv_folds": cv_folds,
        "random_state": RANDOM_STATE,
        "training_rows": int(len(train_frame)),
        "features": FEATURE_COLUMNS,
        "scikit_learn_version": sklearn.__version__,
        "model_sha256": sha256_file(model_path),
    }
    report_path = atomic_json_dump(report, training_dir / "training_report.json")
    return {
        **metadata,
        "model_path": str(model_path),
        "model_comparison_path": str(comparison_path),
        "training_report_path": str(report_path),
        "selected_model": selected_name,
    }
