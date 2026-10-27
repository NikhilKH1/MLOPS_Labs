"""
Load the model saved by main.py and use it to predict held-out test samples.

Run it in a new container with the same volume mounted. If it works, the model
was kept after the container that trained it was removed:
    docker run --rm -v "$(pwd)/output:/app/output" breast-cancer-trainer:v1 python predict.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import joblib

from main import CLASS_NAMES, load_data, split_data


def main():
    parser = argparse.ArgumentParser(description="Predict with the saved breast cancer model.")
    parser.add_argument("-n", "--num-samples", type=int, default=10,
                        help="number of test samples to predict (default: 10)")
    args = parser.parse_args()

    output_dir = Path(os.environ.get("OUTPUT_DIR", "output"))
    model_path = output_dir / "model.joblib"
    if not model_path.exists():
        sys.exit(f"No model found at {model_path}. Run main.py first with the same volume mounted.")

    model = joblib.load(model_path)
    report = json.loads((output_dir / "metrics.json").read_text())
    print(f"Loaded {report['model']} trained at {report['trained_at']} "
          f"in container {report['environment']['hostname']}")

    # Rebuild the same test split the model was evaluated on, using the saved config
    X, y, _ = load_data()
    _, X_test, _, y_test = split_data(X, y, report["config"])
    X_sample, y_sample = X_test[: args.num_samples], y_test[: args.num_samples]

    predictions = model.predict(X_sample)
    p_malignant = model.predict_proba(X_sample)[:, 1]

    print(f"\n{'#':>3}  {'actual':<10} {'predicted':<10} {'P(malignant)':>12}")
    for i, (actual, pred, prob) in enumerate(zip(y_sample, predictions, p_malignant)):
        flag = "" if actual == pred else "  <- wrong"
        print(f"{i:>3}  {CLASS_NAMES[actual]:<10} {CLASS_NAMES[pred]:<10} {prob:>12.3f}{flag}")
    correct = int((predictions == y_sample).sum())
    print(f"\n{correct}/{len(y_sample)} correct")


if __name__ == "__main__":
    main()
