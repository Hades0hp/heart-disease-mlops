# Heart Disease Predictor (MLOps Assignment)

This repository is a refactor of the original notebook into a reproducible, script-first project with MLflow experiment tracking.

## Quickstart (local)

### 1) Create env + install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Get the dataset
**Option A (recommended):** download from UCI automatically and write a CSV:
```bash
python scripts/download_data.py
```

This will create:
- `data/raw/heart_disease_uci.csv`

**Option B:** If you already have a CSV, place it at:
- `data/raw/heart_disease_uci.csv`

### 3) Start MLflow UI (tracking server)
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```
Open: http://127.0.0.1:5000

### 4) Train models (2 models + MLflow logging)
```bash
python -m src.train \
  --data-path data/raw/heart_disease_uci.csv \
  --experiment "Heart Disease Prediction" \
  --cv-splits 5
```

Outputs:
- MLflow runs with params/metrics/artifacts/models
- `reports/` contains saved evaluation artifacts for quick access

## What’s logged to MLflow
- Parameters: preprocessing choices, hyperparameters, CV settings
- Metrics: accuracy, precision, recall, f1, roc_auc (CV and test)
- Artifacts: confusion matrix, ROC curve, classification report, best params JSON
- Model: sklearn Pipeline (preprocess + estimator)

## Project structure
- `src/` core code (train, data loading, evaluation utilities)
- `scripts/` utilities (download data)
- `reports/` exported plots/reports (also logged to MLflow)
