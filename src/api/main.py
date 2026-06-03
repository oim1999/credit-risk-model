"""
FastAPI Application for Credit Risk Scoring

Author: Bati Bank Analytics Team
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.api.pydantic_models import TransactionFeatures, RiskPredictionResponse
from src.predict import predict_risk_probability, predict_credit_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Real-time credit risk scoring for BNPL applications",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: Load model from MLflow registry on startup
model = None


@app.on_event("startup")
async def load_model():
    """
    Load the best model from MLflow registry at startup.
    """
    global model
    try:
        # model = mlflow.sklearn.load_model("models:/credit-risk-model/Production")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None


@app.get("/")
async def root():
    return {
        "message": "Bati Bank Credit Risk API",
        "version": "1.0.0",
        "status": "operational" if model else "model not loaded"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=RiskPredictionResponse)
async def predict_risk(features: TransactionFeatures):
    """
    Predict credit risk for a new customer.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    # Convert to DataFrame
    input_df = pd.DataFrame([features.dict()])

    # Predict
    risk_prob = predict_risk_probability(model, input_df)[0]
    credit_score = predict_credit_score(np.array([risk_prob]))[0]

    # Categorize risk
    if risk_prob < 0.3:
        risk_category = "Low"
    elif risk_prob < 0.7:
        risk_category = "Medium"
    else:
        risk_category = "High"

    return RiskPredictionResponse(
        customer_id=features.customer_id,
        risk_probability=float(risk_prob),
        credit_score=int(credit_score),
        risk_category=risk_category,
        model_version="1.0.0",
    )
