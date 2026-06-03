"""
Pydantic Models for API Request/Response Validation

Author: Bati Bank Analytics Team
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class TransactionFeatures(BaseModel):
    """
    Input features for a single customer transaction profile.
    """
    customer_id: str = Field(..., description="Unique customer identifier")
    total_transaction_amount: float = Field(..., description="Sum of all transaction amounts")
    average_transaction_amount: float = Field(..., description="Average transaction amount")
    transaction_count: int = Field(..., description="Number of transactions")
    std_transaction_amount: float = Field(..., description="Standard deviation of transaction amounts")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of transaction")
    transaction_day: int = Field(..., ge=1, le=31, description="Day of month")
    transaction_month: int = Field(..., ge=1, le=12, description="Month of transaction")
    transaction_year: int = Field(..., description="Year of transaction")
    product_category: str = Field(..., description="Product category")
    channel_id: str = Field(..., description="Transaction channel")
    pricing_strategy: str = Field(..., description="Pricing strategy category")
    recency: int = Field(..., description="Days since last transaction")
    frequency: int = Field(..., description="Number of transactions")
    monetary: float = Field(..., description="Total monetary value")


class RiskPredictionResponse(BaseModel):
    """
    Response model for risk prediction endpoint.
    """
    customer_id: str
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of high risk")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score derived from risk")
    risk_category: str = Field(..., description="Categorized risk level: Low, Medium, High")
    model_version: str = Field(..., description="Version of the model used for prediction")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
