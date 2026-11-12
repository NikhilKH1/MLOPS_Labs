"""Run the complete local experiment: python train.py."""
import json
from src.experiment import train

if __name__ == "__main__":
    table, summary = train()
    print(table.drop(columns="run_id").to_string(index=False))
    print(json.dumps(summary, indent=2))
