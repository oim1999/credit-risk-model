"""
Model Training & Experiment Tracking

Trains multiple classifiers, logs experiments to MLflow, and registers
the best-performing model.

Author: Bati Bank Analytics Team
"""

import os
import sys
import warnings
import logging
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
PROCESSED_DATA_PATH = "data/processed/customer_features.csv"
EXPERIMENT_NAME = "bati-bank-credit-risk"
MODEL_REGISTRY_NAME = "credit-risk-model"


def load_processed_data(path: str = PROCESSED_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the processed customer-level dataset from Task 3/4.
    Returns X (features) and y (is_high_risk target).
    """
    if not os.path.exists(path):
        logger.error(f"Processed data not found at {path}. Run data_processing.py first.")
        sys.exit(1)

    df = pd.read_csv(path)
    logger.info(f"Loaded processed data: {df.shape}")

    # Columns to drop from features
    drop_cols = ["CustomerId", "cluster", "is_high_risk"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    y = df["is_high_risk"]
    X = df.drop(columns=drop_cols)

    # Drop any remaining non-numeric columns (e.g., categorical dominant labels as strings)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(f"Dropping non-numeric columns from X: {non_numeric}")
        X = X.drop(columns=non_numeric)

    logger.info(f"Feature matrix X: {X.shape}, Target y: {y.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


def prepare_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train-test split to preserve class imbalance ratio.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_model(
    model, model_name: str, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """
    Generate predictions and compute all required metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    logger.info(f"{model_name} metrics: {metrics}")
    logger.info(f"\nClassification Report for {model_name}:\n{classification_report(y_test, y_pred)}")

    return metrics


def train_and_log_model(
    model,
    model_name: str,
    param_grid: Dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[object, Dict[str, float], str]:
    """
    Train with GridSearchCV, log parameters/metrics/artifacts to MLflow,
    and return the fitted model, metrics, and run ID.
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_type", model_name)

        # Hyperparameter tuning
        grid = GridSearchCV(
            model,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Log parameters
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("best_cv_roc_auc", grid.best_score_)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Evaluate
        metrics = evaluate_model(best_model, model_name, X_test, y_test)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Log model artifact
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Get run ID for potential registration
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run {run_id} complete for {model_name}")

    return best_model, metrics, run_id


def register_best_model(results: List[Tuple[str, Dict, str]]) -> None:
    """
    Identify the best model by ROC-AUC and register it to the MLflow Model Registry.
    """
    best_name, best_metrics, best_run_id = max(results, key=lambda x: x[1]["roc_auc"])
    model_uri = f"runs:/{best_run_id}/model"

    logger.info(f"Best model: {best_name} (ROC-AUC = {best_metrics['roc_auc']:.4f})")
    logger.info(f"Registering to MLflow Model Registry as '{MODEL_REGISTRY_NAME}'...")

    mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)

    # Log a summary run
    with mlflow.start_run(run_name="Best_Model_Summary"):
        mlflow.log_param("best_model_name", best_name)
        mlflow.log_metrics(best_metrics)
        mlflow.set_tag("registered_model", MODEL_REGISTRY_NAME)

    logger.info("Registration complete.")


def main():
    """
    Main training workflow.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)

    results = []

    # -------------------------------------------------
    # Model 1: Logistic Regression (Baseline - Interpretable)
    # -------------------------------------------------
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    lr_params = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
    }
    lr_model, lr_metrics, lr_run = train_and_log_model(
        lr, "LogisticRegression", lr_params, X_train, X_test, y_train, y_test
    )
    results.append(("LogisticRegression", lr_metrics, lr_run))

    # -------------------------------------------------
    # Model 2: Random Forest
    # -------------------------------------------------
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5],
    }
    rf_model, rf_metrics, rf_run = train_and_log_model(
        rf, "RandomForest", rf_params, X_train, X_test, y_train, y_test
    )
    results.append(("RandomForest", rf_metrics, rf_run))

    # -------------------------------------------------
    # Model 3: Gradient Boosting (XGBoost preferred, fallback to sklearn)
    # -------------------------------------------------
    try:
        from xgboost import XGBClassifier

        logger.info("Training XGBoost...")
        xgb = XGBClassifier(
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        xgb_params = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
        }
        xgb_model, xgb_metrics, xgb_run = train_and_log_model(
            xgb, "XGBoost", xgb_params, X_train, X_test, y_train, y_test
        )
        results.append(("XGBoost", xgb_metrics, xgb_run))
    except ImportError:
        logger.warning("XGBoost not installed; using sklearn GradientBoostingClassifier.")
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
        gb_params = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
        }
        gb_model, gb_metrics, gb_run = train_and_log_model(
            gb, "GradientBoosting", gb_params, X_train, X_test, y_train, y_test
        )
        results.append(("GradientBoosting", gb_metrics, gb_run))

    # -------------------------------------------------
    # Register best model
    # -------------------------------------------------
    register_best_model(results)

    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    comparison = pd.DataFrame(
        {name: metrics for name, metrics, _ in results}
    ).T
    logger.info(f"\n{comparison.round(4)}")


if __name__ == "__main__":
    main()