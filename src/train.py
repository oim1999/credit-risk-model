"""
Model Training & Experiment Tracking

This module handles model training, hyperparameter tuning, and MLflow experiment tracking.

Author: Bati Bank Analytics Team
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def prepare_data(df, target_col='is_high_risk', test_size=0.2):
    """
    Split data into train and test sets.
    """
    X = df.drop(columns=[target_col, 'CustomerId'], errors='ignore')
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, params=None):
    """
    Train a model and log to MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        if params:
            mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(f"{model_name} metrics: {metrics}")

        return model, metrics


def main():
    """
    Main training workflow.
    """
    # TODO: Load processed data, train multiple models, register best model
    pass


if __name__ == "__main__":
    main()
