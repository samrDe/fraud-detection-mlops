"""Train an imbalanced fraud classifier and log reproducible MLflow results."""
from pathlib import Path
import json
import yaml
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from imblearn.over_sampling import RandomOverSampler
from src.data import FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())

def train():
    train_df = pd.read_csv(ROOT / "data/processed/train.csv")
    test_df = pd.read_csv(ROOT / "data/processed/test.csv")
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # Resample training data only. Test distribution remains untouched.
    sampler = RandomOverSampler(
        sampling_strategy=PARAMS["sampling_strategy"],
        random_state=PARAMS["random_state"]
    )
    X_res, y_res = sampler.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=PARAMS["n_estimators"],
        max_depth=PARAMS["max_depth"],
        min_samples_leaf=PARAMS["min_samples_leaf"],
        class_weight="balanced_subsample",
        random_state=PARAMS["random_state"],
        n_jobs=-1
    )

    mlflow.set_experiment("credit-card-fraud-detection")
    with mlflow.start_run() as run:
        model.fit(X_res, y_res)
        probabilities = model.predict_proba(X_test)[:, 1]
        threshold = float(PARAMS["threshold"])
        predictions = (probabilities >= threshold).astype(int)
        metrics = {
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "threshold": threshold,
            "test_fraud_rate": float(y_test.mean()),
        }
        mlflow.log_params({**PARAMS, "model": "RandomForestClassifier",
                           "resampling": "RandomOverSampler"})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_param("train_rows_before_resampling", len(X_train))
        mlflow.log_param("train_rows_after_resampling", len(X_res))

        ROOT.joinpath("artifacts").mkdir(exist_ok=True)
        ROOT.joinpath("reports").mkdir(exist_ok=True)
        artifact_path = ROOT / "artifacts/model.joblib"
        joblib.dump({"model": model, "features": FEATURES, "threshold": threshold}, artifact_path)
        ROOT.joinpath("reports/metrics.json").write_text(json.dumps(metrics, indent=2))

        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        fig.tight_layout()
        cm_path = ROOT / "reports/confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
        fig.tight_layout()
        pr_path = ROOT / "reports/precision_recall_curve.png"
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)

        mlflow.log_artifact(str(artifact_path))
        mlflow.log_artifact(str(ROOT / "reports/metrics.json"))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(pr_path))
        mlflow.sklearn.log_model(model, artifact_path="random_forest")
        print(json.dumps({"run_id": run.info.run_id, **metrics}, indent=2))
    return metrics

if __name__ == "__main__":
    train()
