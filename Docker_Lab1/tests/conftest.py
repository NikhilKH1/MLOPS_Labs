import sys
from pathlib import Path

import pytest

# In the container the app code sits next to tests/ (in /app); in the repo it is in src/
ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "src" if (ROOT / "src").is_dir() else ROOT
sys.path.insert(0, str(APP_DIR))

from main import train  # noqa: E402


@pytest.fixture(scope="session")
def app_dir():
    return APP_DIR


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    """Run the real training job once per test session (all four candidates, 3 folds to keep it quick)."""
    directory = tmp_path_factory.mktemp("artifacts")
    train({
        "model_type": "auto",
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 3,
        "output_dir": str(directory),
    })
    return directory
