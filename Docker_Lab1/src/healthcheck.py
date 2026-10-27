"""
Docker HEALTHCHECK for the dashboard container.

The container is reported healthy only when both are true:
  1. the trainer's artifacts (model.joblib, metrics.json) are in the shared volume
  2. the Streamlit server answers its built-in /_stcore/health endpoint
It uses only the standard library so each check starts fast.
Exit code 0 = healthy, 1 = unhealthy (what Docker expects).
"""
import os
import sys
import urllib.request
from pathlib import Path

ARTIFACTS = ("model.joblib", "metrics.json")


def main():
    output_dir = Path(os.environ.get("OUTPUT_DIR", "output"))
    missing = [name for name in ARTIFACTS if not (output_dir / name).exists()]
    if missing:
        sys.exit(f"unhealthy: missing {', '.join(missing)} in {output_dir}")

    port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
    url = f"http://localhost:{port}/_stcore/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            body = response.read().decode().strip()
    except OSError as error:
        sys.exit(f"unhealthy: {url} not reachable ({error})")
    if body != "ok":
        sys.exit(f"unhealthy: {url} returned {body!r}")
    print("healthy")


if __name__ == "__main__":
    main()
