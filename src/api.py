"""FastAPI inference service. Train src.train before starting the API."""
from pathlib import Path
from typing import Dict
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts/model.joblib"
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")
_bundle = None

class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Exactly the 30 model input features.")

def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not found. Run python -m src.train first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.get("/metadata")
def metadata():
    bundle = get_bundle()
    return {"model": "RandomForestClassifier", "features": bundle["features"],
            "threshold": bundle["threshold"]}

@app.post("/predict")
def predict(request: PredictionRequest):
    bundle = get_bundle()
    expected = bundle["features"]
    supplied = set(request.features)
    missing, extra = sorted(set(expected) - supplied), sorted(supplied - set(expected))
    if missing or extra:
        raise HTTPException(status_code=422, detail={"missing_features": missing, "unexpected_features": extra})
    try:
        row = pd.DataFrame([[request.features[name] for name in expected]], columns=expected)
        if not all(pd.notna(row.iloc[0]).tolist()):
            raise ValueError("Features must be finite numeric values.")
        probability = float(bundle["model"].predict_proba(row)[0, 1])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    threshold = float(bundle["threshold"])
    return {
        "fraud_probability": probability,
        "threshold": threshold,
        "prediction": int(probability >= threshold),
        "label": "fraud" if probability >= threshold else "legitimate",
        "disclaimer": "Model output is a risk score, not a final financial decision."
    }
