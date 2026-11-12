"""Data, model selection, tracking, and artifact generation."""
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.diagnostics import save_diagnostics

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EXPERIMENT = "lab4-diabetes-regression"


def load_data():
    # Avoid the dataset's default full-dataset scaling. Fit scaling inside each fold.
    data = load_diabetes(as_frame=True, scaled=False)
    return data.data, data.target


def split_data():
    X, y = load_data()
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def candidates():
    models = [("linear", LinearRegression())]
    models += [(f"ridge_alpha_{a:g}", Ridge(alpha=a)) for a in (0.1, 1.0, 10.0, 100.0)]
    models += [
        (f"forest_depth_{depth}_leaf_{leaf}", RandomForestRegressor(
            n_estimators=150, max_depth=depth, min_samples_leaf=leaf,
            random_state=SEED, n_jobs=1,
        ))
        for depth in (3, None) for leaf in (1, 5)
    ]
    return [(name, make_pipeline(StandardScaler(), model)) for name, model in models]


def evaluate_cv(pipeline, X, y):
    scores = cross_validate(
        pipeline, X, y, cv=KFold(5, shuffle=True, random_state=SEED),
        scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
        n_jobs=1, error_score="raise",
    )
    return {
        "cv_rmse": float(-scores["test_rmse"].mean()),
        "cv_rmse_std": float(scores["test_rmse"].std()),
        "cv_mae": float(-scores["test_mae"].mean()),
        "cv_r2": float(scores["test_r2"].mean()),
    }


def select_best(rows):
    """Selection deliberately depends only on training-fold validation scores."""
    return min(rows, key=lambda row: row["cv_rmse"])


def configure_tracking(root=ROOT):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT, artifact_location=(root / "mlartifacts").as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def log_model(model, X):
    mlflow.sklearn.log_model(
        model, "model", signature=infer_signature(X, model.predict(X)),
        input_example=X.head(2),
        pip_requirements=str(ROOT / "requirements.txt"),
    )


def plot_comparison(table, output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ordered = table.sort_values("cv_rmse", ascending=False)
    colors = ["#15803d" if value else "#2563eb" for value in ordered.selected]
    axes[0].barh(ordered.candidate, ordered.cv_rmse, xerr=ordered.cv_rmse_std, color=colors)
    axes[0].set(xlabel="Mean validation RMSE (± fold standard deviation)", title="Selection: training-only 5-fold CV")
    axes[1].barh(ordered.candidate, ordered.test_rmse, color=colors)
    axes[1].set(xlabel="Held-out test RMSE", title="Final comparison: never used for selection")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Diabetes regression • green = CV-selected model")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train(root=ROOT):
    root = Path(root).resolve()
    output = root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    configure_tracking(root)
    X_train, X_test, y_train, y_test = split_data()
    rows, fitted = [], {}
    with mlflow.start_run(run_name="comparison") as parent:
        mlflow.set_tags({"assignment": "4", "dataset": "diabetes", "selection": "training_only_5_fold_cv"})
        mlflow.log_params({"random_state": SEED, "test_size": 0.2, "train_rows": len(X_train), "test_rows": len(X_test), "cv_folds": 5, "scaled_dataset": False})
        # Complete all CV selection before inspecting any held-out test metrics.
        for name, pipeline in candidates():
            with mlflow.start_run(run_name=name, nested=True) as run:
                estimator = pipeline.steps[-1][1]
                mlflow.set_tags({"model_type": type(estimator).__name__, "dataset": "diabetes", "phase": "candidate"})
                mlflow.log_params({"model_type": type(estimator).__name__, "preprocessing": "StandardScaler", **estimator.get_params()})
                scores = evaluate_cv(pipeline, X_train, y_train)
                mlflow.log_metrics(scores)
                model = clone(pipeline).fit(X_train, y_train)
                log_model(model, X_train)
                fitted[name] = model
                rows.append({"candidate": name, "model_type": type(estimator).__name__, "run_id": run.info.run_id, **scores})
        best = select_best(rows)
        client = mlflow.MlflowClient()
        for row in rows:
            test_scores = metrics(y_test, fitted[row["candidate"]].predict(X_test))
            row.update({f"test_{key}": value for key, value in test_scores.items()})
            row["selected"] = row["candidate"] == best["candidate"]
            for key, value in test_scores.items():
                client.log_metric(row["run_id"], f"test_{key}", value)
            client.set_tag(row["run_id"], "selected", str(row["selected"]).lower())
        table = pd.DataFrame(rows).sort_values("cv_rmse")
        table.to_csv(output / "model_comparison.csv", index=False)
        plot_comparison(table, output / "model_comparison.png")
        model = fitted[best["candidate"]]
        save_diagnostics(model, X_test, y_test, output, best["candidate"])
        mlflow.log_params({"importance_repeats": 30, "importance_scoring": "neg_root_mean_squared_error", "diagnostics_partition": "held_out_test"})
        model_path = root / "models" / parent.info.run_id
        model_path.parent.mkdir(exist_ok=True)
        mlflow.sklearn.save_model(model, str(model_path), signature=infer_signature(X_train, model.predict(X_train)), pip_requirements=str(ROOT / "requirements.txt"))
        restored = mlflow.sklearn.load_model(str(model_path))
        np.testing.assert_allclose(restored.predict(X_test), model.predict(X_test))
        log_model(model, X_train)
        mlflow.set_tag("best_candidate", best["candidate"])
        mlflow.log_metrics({key: best[key] for key in ("cv_rmse", "test_mae", "test_rmse", "test_r2")})
        summary = {
            "best_candidate": best["candidate"], "selection_metric": "cv_rmse",
            "parent_run_id": parent.info.run_id, "candidate_run_id": best["run_id"],
            "model_path": str(model_path.relative_to(root)),
            "model_uri": f"runs:/{parent.info.run_id}/model",
            "metrics": {key: best[key] for key in ("cv_rmse", "test_mae", "test_rmse", "test_r2")},
            "example_prediction": float(restored.predict(X_test.iloc[:1])[0]),
            "example_target": float(y_test.iloc[0]),
            "train_rows": len(X_train), "test_rows": len(X_test), "candidate_count": len(rows),
        }
        (output / "best_model.json").write_text(json.dumps(summary, indent=2) + "\n")
        X_test.iloc[:1].to_json(output / "example_input.json", orient="records", indent=2)
        mlflow.log_artifacts(str(output), artifact_path="comparison")
    return table, summary


def predict_example(root=ROOT, input_path=None):
    root = Path(root)
    summary_path = root / "outputs" / "best_model.json"
    if not summary_path.exists():
        raise FileNotFoundError("No trained model metadata. Run python train.py first.")
    summary = json.loads(summary_path.read_text())
    model_path = root / summary["model_path"]
    if not model_path.exists():
        raise FileNotFoundError("Saved model is missing. Run python train.py on this machine first.")
    X, _ = load_data()
    if input_path is None:
        frame = split_data()[1].iloc[:1]
    else:
        frame = pd.DataFrame(json.loads(Path(input_path).read_text()))
    if set(frame.columns) != set(X.columns) or frame.empty:
        raise ValueError(f"Input must contain nonempty records with exactly these features: {list(X.columns)}")
    frame = frame.loc[:, X.columns].astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("All input features must be finite numbers.")
    return mlflow.sklearn.load_model(str(model_path)).predict(frame)
