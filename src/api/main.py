"""
FastAPI Application for Credit Risk Scoring

Loads the best model from MLflow registry and serves real-time predictions.

Author: Bati Bank Analytics Team
"""

import os
import logging
from typing import List

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.pydantic_models import TransactionFeatures, RiskPredictionResponse, BatchPredictionRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Real-time credit risk scoring for BNPL applications",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_URI = os.getenv("MODEL_URI", "models:/credit-risk-model/Production")
MODEL_VERSION = "1.0.0"
MIN_SCORE = 300
MAX_SCORE = 850

model = None


def load_model():
    """
    Load the registered model from MLflow.
    """
    global model
    try:
        logger.info(f"Loading model from MLflow: {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        # Fallback: try loading from local artifact
        local_path = "mlruns/0/best_model"
        if os.path.exists(local_path):
            model = mlflow.sklearn.load_model(local_path)
            logger.info("Loaded model from local fallback path.")
        else:
            logger.warning("No model available. Predictions will fail.")


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/")
async def root():
    return {
        "message": "Bati Bank Credit Risk API",
        "version": "1.0.0",
        "status": "operational" if model is not None else "model not loaded",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
    }


def predict_risk_probability(features_df: pd.DataFrame) -> np.ndarray:
    """
    Predict risk probability using the loaded model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return model.predict_proba(features_df)[:, 1]


def calculate_credit_score(risk_probabilities: np.ndarray) -> np.ndarray:
    """
    Convert risk probability to credit score (inverse mapping).
    """
    scores = MIN_SCORE + (1 - risk_probabilities) * (MAX_SCORE - MIN_SCORE)
    return np.round(scores).astype(int)


def categorize_risk(risk_probability: float) -> str:
    """
    Map probability to risk category.
    """
    if risk_probability < 0.3:
        return "Low"
    elif risk_probability < 0.7:
        return "Medium"
    else:
        return "High"


def features_to_dataframe(features: TransactionFeatures) -> pd.DataFrame:
    """
    Convert Pydantic model to pandas DataFrame for model input.
    """
    data = features.dict()
    customer_id = data.pop("customer_id")
    df = pd.DataFrame([data])
    return df, customer_id


@app.post("/predict", response_model=RiskPredictionResponse)
async def predict_risk(features: TransactionFeatures):
    """
    Predict credit risk for a single customer.
    """
    input_df, customer_id = features_to_dataframe(features)

    risk_prob = float(predict_risk_probability(input_df)[0])
    credit_score = int(calculate_credit_score(np.array([risk_prob]))[0])
    risk_category = categorize_risk(risk_prob)

    return RiskPredictionResponse(
        customer_id=customer_id,
        risk_probability=risk_prob,
        credit_score=credit_score,
        risk_category=risk_category,
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=List[RiskPredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict credit risk for multiple customers in one request.
    """
    if not request.customers:
        raise HTTPException(status_code=400, detail="No customers provided")

    records = []
    customer_ids = []

    for cust in request.customers:
        data = cust.dict()
        customer_ids.append(data.pop("customer_id"))
        records.append(data)

    input_df = pd.DataFrame(records)
    risk_probs = predict_risk_probability(input_df)
    credit_scores = calculate_credit_score(risk_probs)

    responses = []
    for cid, prob, score in zip(customer_ids, risk_probs, credit_scores):
        responses.append(
            RiskPredictionResponse(
                customer_id=cid,
                risk_probability=float(prob),
                credit_score=int(score),
                risk_category=categorize_risk(prob),
                model_version=MODEL_VERSION,
            )
        )

    return responses


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)