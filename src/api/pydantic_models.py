"""
Pydantic Models for API Request/Response Validation

Author: Bati Bank Analytics Team
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class TransactionFeatures(BaseModel):
    """
    Input features for a single customer transaction profile.
    Must match the exact feature columns expected by the trained model.
    """
    customer_id: str = Field(..., description="Unique customer identifier")

    # Aggregate features
    total_transaction_amount: float = Field(..., description="Sum of all transaction amounts")
    average_transaction_amount: float = Field(..., description="Average transaction amount")
    transaction_count: int = Field(..., ge=0, description="Number of transactions")
    std_transaction_amount: float = Field(..., description="Std dev of transaction amounts")
    amount_range: float = Field(..., description="Max - Min transaction amount")
    amount_cv: float = Field(..., description="Coefficient of variation")
    log1p_amount_sum: float = Field(..., description="Log-transformed total amount")
    credit_ratio: float = Field(..., ge=0, le=1, description="Ratio of credit transactions")
    avg_abs_amount: float = Field(..., description="Average absolute transaction amount")
    max_single_value: float = Field(..., description="Maximum single transaction value")

    # Temporal features
    preferred_hour: int = Field(..., ge=0, le=23, description="Most frequent transaction hour")
    hour_std: float = Field(..., ge=0, description="Std dev of transaction hours")
    weekend_ratio: float = Field(..., ge=0, le=1, description="Fraction of weekend transactions")
    unique_days: int = Field(..., ge=1, description="Unique days with transactions")
    unique_months: int = Field(..., ge=1, description="Unique months with transactions")

    # RFM features
    recency: int = Field(..., ge=0, description="Days since last transaction")
    frequency: int = Field(..., ge=1, description="Number of transactions")
    monetary: float = Field(..., ge=0, description="Total monetary value")
    log_monetary: float = Field(..., description="Log-transformed monetary value")

    # Categorical dominant
    product_category_dominant: str = Field(..., description="Most frequent product category")
    channel_id_dominant: str = Field(..., description="Most frequent channel")
    pricing_strategy_dominant: str = Field(..., description="Most frequent pricing strategy")
    provider_id_dominant: str = Field(..., description="Most frequent provider")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "C1",
                "total_transaction_amount": 3000.0,
                "average_transaction_amount": 1000.0,
                "transaction_count": 3,
                "std_transaction_amount": 500.0,
                "amount_range": 1500.0,
                "amount_cv": 0.5,
                "log1p_amount_sum": 8.0,
                "credit_ratio": 0.0,
                "avg_abs_amount": 1000.0,
                "max_single_value": 2000.0,
                "preferred_hour": 14,
                "hour_std": 2.0,
                "weekend_ratio": 0.0,
                "unique_days": 3,
                "unique_months": 1,
                "recency": 5,
                "frequency": 3,
                "monetary": 3000.0,
                "log_monetary": 8.0,
                "product_category_dominant": "airtime",
                "channel_id_dominant": "ChannelId_3",
                "pricing_strategy_dominant": "2",
                "provider_id_dominant": "ProviderId_4"
            }
        }


class RiskPredictionResponse(BaseModel):
    """
    Response model for the /predict endpoint.
    """
    customer_id: str
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of high risk")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score derived from risk")
    risk_category: str = Field(..., description="Risk level: Low, Medium, High")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """
    Optional: Batch prediction request for multiple customers.
    """
    customers: List[TransactionFeatures]