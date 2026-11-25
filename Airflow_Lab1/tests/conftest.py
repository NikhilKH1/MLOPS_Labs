from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 320
    latitude = rng.uniform(32.5, 41.8, rows)
    longitude = rng.uniform(-124.5, -114.2, rows)
    med_inc = rng.lognormal(mean=1.25, sigma=0.45, size=rows)
    house_age = rng.integers(1, 53, rows).astype(float)
    ave_rooms = rng.uniform(2.0, 9.0, rows)
    ave_bedrooms = rng.uniform(0.7, 2.2, rows)
    population = rng.integers(50, 6000, rows).astype(float)
    ave_occupancy = rng.uniform(1.0, 6.0, rows)
    target = (
        0.42 * med_inc
        + 0.03 * ave_rooms
        - 0.05 * ave_bedrooms
        - 0.025 * (latitude - 36.0)
        - 0.012 * (longitude + 119.0)
        + rng.normal(0, 0.18, rows)
    )
    target = np.clip(target, 0.15, None)
    return pd.DataFrame(
        {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrooms,
            "Population": population,
            "AveOccup": ave_occupancy,
            "Latitude": latitude,
            "Longitude": longitude,
            "MedHouseVal": target,
        }
    )
