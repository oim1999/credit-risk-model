"""
Unit Tests for Data Processing Module

Author: Bati Bank Analytics Team
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    AggregateFeatureEngineer,
    TemporalFeatureEngineer,
    RFMCalculator,
    RiskLabelAssigner,
    DataCleaner,
)


class TestAggregateFeatureEngineer:
    """Test suite for aggregate feature engineering."""

    def test_expected_columns(self):
        """Test that the engineer returns expected columns."""
        sample_data = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C2'],
            'Amount': [100.0, 200.0, 50.0, 150.0],
            'Value': [100.0, 200.0, 50.0, 150.0],
            'TransactionStartTime': pd.to_datetime([
                '2018-11-15', '2018-11-16', '2018-11-15', '2018-11-17'
            ])
        })

        engineer = AggregateFeatureEngineer()
        result = engineer.fit_transform(sample_data)

        expected_cols = [
            'CustomerId', 'Amount_sum', 'Amount_mean', 'Amount_std',
            'Amount_count', 'Amount_max', 'Amount_min',
            'Value_sum', 'Value_mean', 'Value_std', 'Value_max', 'Value_min',
            'amount_range', 'value_range', 'amount_cv', 'value_cv',
            'log1p_Amount_sum', 'log1p_Value_sum', 'credit_ratio', 'avg_abs_amount',
            'max_single_value'
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_single_customer_aggregation(self):
        """Test aggregation logic for a single customer."""
        sample_data = pd.DataFrame({
            'CustomerId': ['C1', 'C1'],
            'Amount': [100.0, 200.0],
            'Value': [100.0, 200.0],
            'TransactionStartTime': pd.to_datetime(['2018-11-15', '2018-11-16'])
        })

        engineer = AggregateFeatureEngineer()
        result = engineer.fit_transform(sample_data)

        assert result['Amount_sum'].iloc[0] == 300.0
        assert result['Amount_mean'].iloc[0] == 150.0
        assert result['Amount_count'].iloc[0] == 2


class TestTemporalFeatureEngineer:
    """Test suite for temporal feature engineering."""

    def test_temporal_features_created(self):
        """Test that temporal features are correctly extracted and aggregated."""
        sample_data = pd.DataFrame({
            'CustomerId': ['C1'],
            'TransactionStartTime': pd.to_datetime(['2018-11-15 14:30:00']),
            'Amount': [100.0]  # included so groupby works
        })

        engineer = TemporalFeatureEngineer()
        result = engineer.fit_transform(sample_data)

        # Result is customer-level aggregated
        assert 'CustomerId' in result.columns
        assert 'preferred_hour' in result.columns
        assert 'hour_std' in result.columns
        assert 'weekend_ratio' in result.columns
        assert 'unique_days' in result.columns
        assert 'unique_months' in result.columns
        assert result['preferred_hour'].iloc[0] == 14
        assert result['unique_months'].iloc[0] == 1  # Only November

    def test_weekend_flag(self):
        """Test weekend detection logic."""
        sample_data = pd.DataFrame({
            'CustomerId': ['C1', 'C1'],
            'TransactionStartTime': pd.to_datetime([
                '2018-11-17 10:00:00',  # Saturday
                '2018-11-18 10:00:00'   # Sunday
            ]),
            'Amount': [100.0, 200.0]
        })

        engineer = TemporalFeatureEngineer()
        result = engineer.fit_transform(sample_data)

        # weekend_ratio = 2/2 = 1.0 (both transactions on weekend)
        assert result['weekend_ratio'].iloc[0] == 1.0


class TestRFMCalculator:
    """Test suite for RFM calculation."""

    def test_rfm_values(self):
        """Test RFM metric calculation with Value column present."""
        sample_data = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C1'],
            'Amount': [100.0, 200.0, 300.0],
            'Value': [100.0, 200.0, 300.0],
            'TransactionStartTime': pd.to_datetime([
                '2018-11-15', '2018-11-20', '2018-11-25'
            ])
        })

        calculator = RFMCalculator(snapshot_date='2018-11-30')
        result = calculator.fit_transform(sample_data)

        assert result['recency'].iloc[0] == 5  # Days from Nov 25 to Nov 30
        assert result['frequency'].iloc[0] == 3
        assert result['monetary'].iloc[0] == 600.0
        assert result['log_monetary'].iloc[0] == pytest.approx(np.log1p(600.0), 0.001)

    def test_rfm_with_single_transaction(self):
        """Test RFM when customer has only one transaction."""
        sample_data = pd.DataFrame({
            'CustomerId': ['C1'],
            'Amount': [500.0],
            'Value': [500.0],
            'TransactionStartTime': pd.to_datetime(['2018-11-20'])
        })

        calculator = RFMCalculator(snapshot_date='2018-11-25')
        result = calculator.fit_transform(sample_data)

        assert result['recency'].iloc[0] == 5
        assert result['frequency'].iloc[0] == 1
        assert result['monetary'].iloc[0] == 500.0


class TestRiskLabelAssigner:
    """Test suite for risk label assignment."""

    def test_high_risk_label_created(self):
        """Test that is_high_risk column is created with binary values."""
        sample_data = pd.DataFrame({
            'CustomerId': ['C1', 'C2', 'C3', 'C4', 'C5'],
            'recency': [1, 5, 30, 60, 90],
            'frequency': [50, 20, 10, 5, 2],
            'log_monetary': [8.0, 6.0, 4.0, 2.0, 1.0]
        })

        assigner = RiskLabelAssigner(n_clusters=3, random_state=42)
        result = assigner.fit_transform(sample_data)

        assert 'is_high_risk' in result.columns
        assert result['is_high_risk'].isin([0, 1]).all()
        assert result['cluster'].nunique() == 3

    def test_high_risk_cluster_identified(self):
        """Test that the highest recency + lowest monetary cluster is labeled high risk."""
        sample_data = pd.DataFrame({
            'CustomerId': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
            'recency': [100, 100, 100, 1, 1, 1],
            'frequency': [1, 1, 1, 50, 50, 50],
            'log_monetary': [1.0, 1.0, 1.0, 8.0, 8.0, 8.0]
        })

        assigner = RiskLabelAssigner(n_clusters=2, random_state=42)
        result = assigner.fit_transform(sample_data)

        # The 3 customers with high recency + low monetary should be high risk
        high_risk_customers = result[result['is_high_risk'] == 1]['CustomerId'].tolist()
        assert 'C1' in high_risk_customers or 'C2' in high_risk_customers or 'C3' in high_risk_customers


class TestDataCleaner:
    """Test suite for data cleaning."""

    def test_constant_columns_dropped(self):
        """Test that constant columns are automatically dropped."""
        sample_data = pd.DataFrame({
            'CustomerId': ['C1', 'C2', 'C3'],
            'Amount': [100.0, 200.0, 300.0],
            'CountryCode': [256, 256, 256]  # Constant column
        })

        cleaner = DataCleaner()
        result = cleaner.fit_transform(sample_data)

        assert 'CountryCode' not in result.columns
        assert 'Amount' in result.columns

    def test_infinities_replaced(self):
        """Test that infinite values are replaced."""
        sample_data = pd.DataFrame({
            'Amount': [100.0, np.inf, -np.inf, 200.0]
        })

        cleaner = DataCleaner(drop_constant=False)
        result = cleaner.fit_transform(sample_data)

        assert not np.isinf(result['Amount']).any()
        assert result['Amount'].isnull().sum() == 2  # inf replaced with NaN