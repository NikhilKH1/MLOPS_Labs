"""Load the exported pipeline and predict raw-feature JSON records."""
import argparse
import json
from src.experiment import predict_example

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON list of records containing all ten raw features")
    args = parser.parse_args()
    try:
        predictions = predict_example(input_path=args.input)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")
    print(json.dumps({"predictions": predictions.tolist()}))
