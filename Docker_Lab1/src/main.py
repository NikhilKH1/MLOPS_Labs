"""
Docker Lab 1 (modified): train a breast cancer classifier inside a container.

What changed from the original lab (Iris + a fixed RandomForest):
  - Breast Cancer Wisconsin dataset (569 samples, 30 features, binary target)
  - four candidate models compared with stratified k-fold cross-validation
  - settings come from environment variables, so `docker run -e` can change them
  - the model and a metrics report are written to OUTPUT_DIR, which is meant to
    be mounted as a volume so they are still there after the container exits
"""
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Name -> function that builds the estimator for a given random seed
CANDIDATES = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=5000, random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed),
    "gradient_boosting": lambda seed: GradientBoostingClassifier(random_state=seed),
    "svm": lambda seed: SVC(kernel="rbf", probability=True, random_state=seed),
}

CLASS_NAMES = ["benign", "malignant"]


def load_config():
    """Read training settings from environment variables (defaults set in the Dockerfile)."""
    config = {
        "model_type": os.environ.get("MODEL_TYPE", "auto"),
        "test_size": float(os.environ.get("TEST_SIZE", "0.2")),
        "random_state": int(os.environ.get("RANDOM_STATE", "42")),
        "cv_folds": int(os.environ.get("CV_FOLDS", "5")),
        "output_dir": os.environ.get("OUTPUT_DIR", "output"),
    }
    valid = ["auto", *CANDIDATES]
    if config["model_type"] not in valid:
        raise SystemExit(f"MODEL_TYPE must be one of {valid}, got {config['model_type']!r}")
    if not 0 < config["test_size"] < 1:
        raise SystemExit(f"TEST_SIZE must be between 0 and 1, got {config['test_size']}")
    if config["cv_folds"] < 2:
        raise SystemExit(f"CV_FOLDS must be at least 2, got {config['cv_folds']}")
    return config


def load_data():
    """
    Load the Breast Cancer Wisconsin dataset.
    sklearn encodes malignant as 0 and benign as 1; the labels are flipped here so
    that 1 = malignant, which makes precision/recall describe the cancer class.
    """
    data = load_breast_cancer()
    return data.data, 1 - data.target, list(data.feature_names)


def split_data(X, y, config):
    """Stratified split so both sets keep the same benign/malignant ratio."""
    return train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_state"], stratify=y
    )


def build_pipeline(name, seed):
    """Scale the features, then fit the classifier; both steps are saved together."""
    return Pipeline([("scaler", StandardScaler()), ("clf", CANDIDATES[name](seed))])


def select_model(X_train, y_train, config):
    """Cross-validate the candidates on the training set and return the best one by ROC-AUC."""
    names = list(CANDIDATES) if config["model_type"] == "auto" else [config["model_type"]]
    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["random_state"])
    cv_scores = {}
    print(f"Cross-validating {len(names)} model(s) with {config['cv_folds']} folds:")
    for name in names:
        scores = cross_val_score(
            build_pipeline(name, config["random_state"]), X_train, y_train, cv=cv, scoring="roc_auc"
        )
        cv_scores[name] = round(float(scores.mean()), 4)
        print(f"  {name:<20} ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
    best = max(cv_scores, key=cv_scores.get)
    return best, cv_scores


def evaluate(model, X_test, y_test):
    """Metrics on the held-out test set, with malignant as the positive class."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }, confusion_matrix(y_test, y_pred).tolist()


if __name__ == "__main__":
    config = load_config()
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features "
          f"({len(X_train)} train / {len(X_test)} test)")

    best_name, cv_scores = select_model(X_train, y_train, config)
    print(f"Selected model: {best_name}")

    # Train the selected model on the full training set and score it on the test set
    model = build_pipeline(best_name, config["random_state"])
    model.fit(X_train, y_train)
    test_metrics, cm = evaluate(model, X_test, y_test)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": best_name,
        "config": config,
        "cv_roc_auc": cv_scores,
        "test_metrics": test_metrics,
        "confusion_matrix": {"labels": CLASS_NAMES, "rows_actual_cols_predicted": cm},
        "dataset": {
            "name": "Breast Cancer Wisconsin (sklearn.datasets.load_breast_cancer)",
            "n_samples": int(len(X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "classes": CLASS_NAMES,
            "features": feature_names,
        },
        # Inside a container the hostname is the container ID
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }

    # Save the model and the report
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    print("Test metrics (positive class = malignant):")
    for metric, value in test_metrics.items():
        print(f"  {metric:<10} {value:.4f}")
    print(f"Confusion matrix [benign, malignant]: {cm}")
    print(f"Saved model.joblib and metrics.json to {output_dir.resolve()}")
    print("The model training was successful")
