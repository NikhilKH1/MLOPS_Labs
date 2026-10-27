"""
Read what the trainer wrote to OUTPUT_DIR, and rebuild the test split it was
evaluated on. predict.py and the dashboard both use these functions, so they
always load the same model and show the same held-out samples.
"""
import json
import os
from pathlib import Path

import joblib

from main import MODEL_FILE, REPORT_FILE, load_data, split_data


def output_dir():
    """Where the artifacts live: the shared volume inside the containers."""
    return Path(os.environ.get("OUTPUT_DIR", "output"))


def missing_artifacts(directory):
    """Names of the expected files that are not in `directory` (empty when training has run)."""
    return [name for name in (MODEL_FILE, REPORT_FILE) if not (Path(directory) / name).exists()]


def load_artifacts(directory):
    """Load the trained pipeline and its metrics report."""
    directory = Path(directory)
    model = joblib.load(directory / MODEL_FILE)
    report = json.loads((directory / REPORT_FILE).read_text())
    return model, report


def load_test_split(config):
    """Recreate the exact held-out test set using the split settings saved in the report."""
    X, y, feature_names = load_data()
    _, X_test, _, y_test = split_data(X, y, config)
    return X_test, y_test, feature_names
