from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import mlflow
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data import load_dataset
from src.features import infer_feature_spec, build_preprocessor
from src.evaluate import (
    compute_metrics,
    save_confusion_matrix,
    save_roc_curve,
    save_classification_report,
    save_json,
)

warnings.filterwarnings("ignore", category=UserWarning)


def train_one(
    *,
    experiment: str,
    run_name: str,
    X,
    y,
    pipeline: Pipeline,
    param_grid: dict,
    cv_splits: int,
    seed: int,
    threshold: float,
    reports_dir: Path,
):
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "dataset": "Heart Disease UCI",
                "cv": f"StratifiedKFold(splits={cv_splits})",
                "threshold": str(threshold),
            }
        )

        # Log "static" params describing pipeline
        mlflow.log_param("model", run_name)
        mlflow.log_param("cv_splits", cv_splits)
        mlflow.log_param("random_seed", seed)

        # GridSearchCV (CV metrics are captured via best_score_)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, 
                            random_state=seed)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid.fit(X, y)

        # Log best params + best CV score
        best_params = grid.best_params_
        mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})
        mlflow.log_metric("cv_best_roc_auc", float(grid.best_score_))

        # Evaluate on training data quickly (for assignment clarity)
        y_proba = grid.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = compute_metrics(y, y_pred, y_proba)
        for k, v in metrics.items():
            mlflow.log_metric(f"train_{k}", v)

        # Save + log artifacts
        run_reports = reports_dir / run_name
        cm_path = save_confusion_matrix(y, y_pred, 
                                        run_reports / "confusion_matrix.png")
        roc_path = save_roc_curve(y, y_proba, run_reports / "roc_curve.png")
        rep_path = save_classification_report(
            y, y_pred, run_reports / "classification_report.txt"
        )
        params_path = save_json(best_params, run_reports / "best_params.json")
        metrics_path = save_json(metrics, run_reports / "train_metrics.json")

        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        mlflow.log_artifact(str(roc_path), artifact_path="plots")
        mlflow.log_artifact(str(rep_path), artifact_path="reports")
        mlflow.log_artifact(str(params_path), artifact_path="reports")
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")

        # Log model (pipeline includes preprocessing)
        mlflow.sklearn.log_model(grid.best_estimator_, name="model")

        return grid.best_estimator_, best_params, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, default="data/raw/heart_disease_uci.csv"
    )
    parser.add_argument("--experiment", type=str, 
                        default="Heart Disease Prediction")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = load_dataset(args.data_path)
    spec = infer_feature_spec(df, target_col="target")

    X = df.drop(columns=[spec.target_col])
    y = df[spec.target_col].astype(int)

    preprocessor = build_preprocessor(spec)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Logistic Regression (tunable C)
    lr = LogisticRegression(max_iter=2000, solver="liblinear")
    lr_pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", lr),
        ]
    )
    lr_grid = {
        "model__C": [0.1, 1.0, 10.0],
        "model__penalty": ["l1", "l2"],
    }

    train_one(
        experiment=args.experiment,
        run_name="LogisticRegression",
        X=X,
        y=y,
        pipeline=lr_pipe,
        param_grid=lr_grid,
        cv_splits=args.cv_splits,
        seed=args.seed,
        threshold=args.threshold,
        reports_dir=reports_dir,
    )

    # Random Forest
    rf = RandomForestClassifier(random_state=args.seed)
    rf_pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", rf),
        ]
    )
    rf_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_split": [2, 5],
    }

    train_one(
        experiment=args.experiment,
        run_name="RandomForest",
        X=X,
        y=y,
        pipeline=rf_pipe,
        param_grid=rf_grid,
        cv_splits=args.cv_splits,
        seed=args.seed,
        threshold=args.threshold,
        reports_dir=reports_dir,
    )


if __name__ == "__main__":
    main()
