"""
Load the model saved by main.py and use it to predict held-out test samples.

Run it in a new container with the same volume mounted. If it works, the model
was kept after the container that trained it was removed:
    docker run --rm -v "$(pwd)/output:/app/output" breast-cancer-trainer:v1 python predict.py
"""
import argparse
import sys

from artifacts import load_artifacts, load_test_split, missing_artifacts, output_dir
from main import CLASS_NAMES, MODEL_FILE


def main():
    parser = argparse.ArgumentParser(description="Predict with the saved breast cancer model.")
    parser.add_argument("-n", "--num-samples", type=int, default=10,
                        help="number of test samples to predict (default: 10)")
    args = parser.parse_args()

    directory = output_dir()
    if missing_artifacts(directory):
        sys.exit(f"No model found at {directory / MODEL_FILE}. "
                 "Run main.py first with the same volume mounted.")

    model, report = load_artifacts(directory)
    print(f"Loaded {report['model']} trained at {report['trained_at']} "
          f"in container {report['environment']['hostname']}")

    # Rebuild the same test split the model was evaluated on, using the saved config
    X_test, y_test, _ = load_test_split(report["config"])
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
