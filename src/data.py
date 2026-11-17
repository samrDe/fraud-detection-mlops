"""Validate the Kaggle dataset and create leakage-safe stratified splits."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/creditcard.csv"
PROCESSED = ROOT / "data/processed"
TARGET = "Class"
FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

def load_and_validate(path=RAW):
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required].copy()
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values; investigate before training.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 (legitimate) and 1 (fraud).")
    if df[TARGET].nunique() != 2:
        raise ValueError("Both target classes must be present.")
    return df

def create_splits(input_path=RAW, output_dir=PROCESSED, test_size=0.20, random_state=42):
    df = load_and_validate(input_path)
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET]
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    # Reference feature sample for drift comparison; no labels are needed by Evidently.
    train[FEATURES].sample(n=min(10000, len(train)), random_state=random_state).to_csv(
        output_dir / "reference.csv", index=False
    )
    return train, test

if __name__ == "__main__":
    create_splits()
    print(f"Created train/test/reference files in {PROCESSED}")
