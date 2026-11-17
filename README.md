# End-to-End Credit Card Fraud Detection with MLOps

A reproducible fraud-detection project demonstrating imbalanced classification, precision-recall evaluation, explainability, experiment tracking, data/model versioning, API serving, containerisation, CI, and drift monitoring.

> **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset is not redistributed here. Download `creditcard.csv` from Kaggle and place it at `data/raw/creditcard.csv`.

## Architecture

```text
Kaggle CSV
   │
   ▼
src/data.py ── schema checks + stratified split ──► data/processed/*.csv
   │                                                   │
   │                                                   ▼
   └────────────────────────────────────────────── src/train.py
                                                       │
                       ┌───────────────────────────────┼────────────────┐
                       ▼                               ▼                ▼
                 MLflow runs                     model.joblib     metrics.json
                       │                               │
                       ▼                               ▼
                 experiment UI                    FastAPI /predict
                                                       │
Reference data + current data ──► Evidently drift report
```

## What this project demonstrates

- **Imbalanced learning:** stratified split and `RandomOverSampler` applied only to the training partition.
- **Evaluation:** average precision, precision, recall, F1, ROC-AUC, confusion matrix, and precision-recall curve. The default decision threshold is configurable.
- **Explainability:** SHAP global feature importance (optional extra install).
- **Experiment tracking:** MLflow logs parameters, metrics, model and plots.
- **Data versioning:** DVC pipeline and `.dvc`-tracked dataset workflow.
- **Serving:** FastAPI health, metadata and prediction endpoints.
- **Deployment:** Docker image and Compose setup.
- **CI:** GitHub Actions installs dependencies, runs tests and builds the Docker image.
- **Monitoring:** Evidently compares reference and current feature distributions and creates an HTML report.

## Project layout

```text
.
├── .github/workflows/ci.yml
├── configs/params.yaml
├── data/raw/.gitkeep
├── data/processed/.gitkeep
├── docs/                         # add screenshots and experiment notes here
├── monitoring/drift_report.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   ├── explain.py
│   └── api.py
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   └── test_api.py
├── artifacts/.gitkeep
├── reports/.gitkeep
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
└── requirements.txt
```

## 1. Setup

Recommended: Python 3.11, Git, and Docker Desktop (for container steps).

```bash
git clone https://github.com/<YOUR-USERNAME>/fraud-detection-mlops.git
cd fraud-detection-mlops

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Get the dataset

1. Create/login to a Kaggle account and accept the dataset terms.
2. Download `creditcard.csv`.
3. Put it in `data/raw/creditcard.csv`.

The target column must be named `Class` (0 = legitimate, 1 = fraud). The feature columns are `Time`, `Amount`, and `V1` through `V28`.

## 3. Run the pipeline

```bash
python -m src.data
python -m src.train
```

Or use DVC to run the stages and cache their outputs:

```bash
dvc init
dvc add data/raw/creditcard.csv
git add data/raw/creditcard.csv.dvc .gitignore
dvc repro
```

The first `dvc repro` runs preprocessing and training. Later runs reuse unchanged stages. To share the dataset with collaborators, configure a DVC remote (S3, Google Drive, Azure, SSH, etc.) and run `dvc push`. Do not commit the raw dataset to Git.

## 4. Explore MLflow

Training logs to a local `mlruns/` directory. In a second terminal:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. Compare runs by average precision, recall, and precision. For a shared/team setup, configure a remote MLflow tracking server and artifact store.

## 5. Run the API

The API expects the 30 numeric model features. Example request:

```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":{"Time":0,"Amount":149.62,"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.33,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02}}'
```

This is a format example, not a known fraud/legitimate transaction. A prediction is a model score, not a financial decision.

## 6. Docker

Train the model locally first so `artifacts/model.joblib` exists, then:

```bash
docker compose up --build
```

API: `http://localhost:8000`; interactive docs: `http://localhost:8000/docs`.

## 7. Explainability

Install SHAP if it is not already available:

```bash
pip install shap
python -m src.explain
```

The script writes a SHAP summary plot to `reports/shap_summary.png`. Explainability output is a model-level diagnostic, not proof of causality.

## 8. Drift monitoring

Provide a reference dataset and a recent/current dataset with the same 30 feature columns. The script does not use the target column.

```bash
python monitoring/drift_report.py \
  --reference data/processed/reference.csv \
  --current data/processed/current.csv \
  --output reports/drift_report.html
```

The report is saved as HTML. A detected feature drift is a prompt to investigate data quality, seasonality, upstream changes and model performance; it does not by itself prove that model accuracy has degraded.

## 9. Tests and CI

```bash
pytest -q
```

GitHub Actions runs the test suite and builds the Docker image on pushes and pull requests to `main`/`master`.

## Important modelling notes

- This dataset is extremely imbalanced. Accuracy is not an appropriate headline metric.
- Oversampling is performed **only after** the train/test split to prevent leakage.
- The test set is left untouched by resampling.
- A threshold of 0.5 is a baseline. In a real fraud operation, choose a threshold using validation data, review capacity, fraud loss and false-positive cost.
- The dataset is anonymised and historical. Strong benchmark results do not guarantee performance on live transactions.
- This repository is a portfolio/demo project, not a production payment-risk system. Add authentication, rate limiting, secrets management, audit logging, privacy controls, model approval and live labelled feedback before real deployment.

## Suggested GitHub portfolio improvements

1. Add screenshots of MLflow runs, Swagger docs, SHAP summary and Evidently report to `docs/`.
2. Record the exact dataset version and your final train/test split seed.
3. Add a short model card with intended use, limitations, threshold rationale and evaluation results.
4. Add a comparison experiment (e.g. Logistic Regression vs Random Forest) and explain the trade-off between recall and false positives.
5. Never upload Kaggle credentials, raw cardholder data, MLflow local state or large model files to public GitHub.

## References

- [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking/)
- [DVC Get Started](https://dvc.org/doc/start)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Evidently](https://docs.evidentlyai.com/)
- [SHAP](https://shap.readthedocs.io/)
