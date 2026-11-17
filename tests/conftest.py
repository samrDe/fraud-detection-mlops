import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
import src.api as api


class DummyModel:
    def predict_proba(self, frame):
        return np.array([[0.9, 0.1]])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_bundle", {
        "model": DummyModel(),
        "features": ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
        "threshold": 0.5
    })
    return TestClient(api.app)
