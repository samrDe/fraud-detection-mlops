import pandas as pd
import pytest
from src.data import FEATURES, load_and_validate

def test_load_and_validate_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Class": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_and_validate(path)

def test_load_and_validate_accepts_valid_schema(tmp_path):
    path = tmp_path / "valid.csv"
    df = pd.DataFrame([{**{c: 0.0 for c in FEATURES}, "Class": 0},
                       {**{c: 1.0 for c in FEATURES}, "Class": 1}])
    df.to_csv(path, index=False)
    loaded = load_and_validate(path)
    assert len(loaded) == 2
    assert set(loaded["Class"]) == {0, 1}
