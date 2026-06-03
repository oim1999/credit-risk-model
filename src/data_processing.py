"""
Data Processing & Feature Engineering Pipeline

Transforms raw Xente transaction-level data into a customer-level,
model-ready dataset with an RFM-based proxy risk label.

Covers:
    - Task 3: Aggregate features, temporal features, encoding, scaling, WoE/IV
    - Task 4: RFM calculation + K-Means clustering for is_high_risk proxy target

Author: Bati Bank Analytics Team
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# =============================================================================
# 1. AGGREGATE FEATURE ENGINEER (Task 3)
# =============================================================================

class AggregateFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Aggregate transaction-level data to customer-level features.
    Computes sums, means, stds, counts, min/max, and ratios.
    """

    def __init__(
        self,
        customer_id_col: str = "CustomerId",
        amount_col: str = "Amount",
        value_col: str = "Value",
        timestamp_col: str = "TransactionStartTime",
    ):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.value_col = value_col
        self.timestamp_col = timestamp_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering aggregate features per customer...")

        df = X.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        # Core aggregates
        agg_specs = {
            self.amount_col: ["sum", "mean", "std", "count", "max", "min"],
            self.value_col: ["sum", "mean", "std", "max", "min"],
        }

        customer_agg = df.groupby(self.customer_id_col).agg(agg_specs).reset_index()

        # Flatten multi-index columns
        customer_agg.columns = [
            f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
            for col in customer_agg.columns.values
        ]

        # Rename customer column back cleanly
        customer_agg.rename(
            columns={f"{self.customer_id_col}_": self.customer_id_col},
            inplace=True,
        )

        # Derived features
        customer_agg["amount_range"] = (
            customer_agg[f"{self.amount_col}_max"] - customer_agg[f"{self.amount_col}_min"]
        )
        customer_agg["value_range"] = (
            customer_agg[f"{self.value_col}_max"] - customer_agg[f"{self.value_col}_min"]
        )

        # Coefficient of variation (handle division by zero)
        customer_agg["amount_cv"] = (
            customer_agg[f"{self.amount_col}_std"] / customer_agg[f"{self.amount_col}_mean"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        customer_agg["value_cv"] = (
            customer_agg[f"{self.value_col}_std"] / customer_agg[f"{self.value_col}_mean"]
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # Log-transformed monetary features (handles skewness > 50 from EDA)
        for col in [f"{self.amount_col}_sum", f"{self.value_col}_sum"]:
            customer_agg[f"log1p_{col}"] = np.log1p(customer_agg[col].abs())

        # Ratio of credit to debit transactions (uses negative Amount)
        credit_mask = df[self.amount_col] < 0
        credit_counts = (
            df[credit_mask].groupby(self.customer_id_col).size()
            .reindex(customer_agg[self.customer_id_col], fill_value=0)
        )
        total_counts = customer_agg[f"{self.amount_col}_count"]
        customer_agg["credit_ratio"] = (credit_counts / total_counts.replace(0, np.nan)).fillna(0)

        # Average transaction value (absolute)
        customer_agg["avg_abs_amount"] = (
            customer_agg[f"{self.amount_col}_sum"].abs() / customer_agg[f"{self.amount_col}_count"]
        )

        # Max single transaction as proxy for "big spender" tail risk
        customer_agg["max_single_value"] = customer_agg[f"{self.value_col}_max"]

        logger.info(f"Aggregate features engineered: {customer_agg.shape[1]} columns")
        return customer_agg


# =============================================================================
# 2. TEMPORAL FEATURE ENGINEER (Task 3)
# =============================================================================

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from transaction timestamps and aggregate
    to customer-level behavioral patterns.
    """

    def __init__(
        self,
        customer_id_col: str = "CustomerId",
        timestamp_col: str = "TransactionStartTime",
    ):
        self.customer_id_col = customer_id_col
        self.timestamp_col = timestamp_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering temporal features...")

        df = X.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        # Transaction-level temporal features
        df["transaction_hour"] = df[self.timestamp_col].dt.hour
        df["transaction_day"] = df[self.timestamp_col].dt.day
        df["transaction_month"] = df[self.timestamp_col].dt.month
        df["transaction_year"] = df[self.timestamp_col].dt.year
        df["transaction_weekday"] = df[self.timestamp_col].dt.weekday
        df["is_weekend"] = df["transaction_weekday"].isin([5, 6]).astype(int)

        # Customer-level temporal aggregates
        temporal_agg = df.groupby(self.customer_id_col).agg(
            preferred_hour=("transaction_hour", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            hour_std=("transaction_hour", "std"),
            weekend_ratio=("is_weekend", "mean"),
            unique_days=("transaction_day", "nunique"),
            unique_months=("transaction_month", "nunique"),
            first_transaction=("transaction_year", "min"),
            last_transaction=("transaction_year", "max"),
        ).reset_index()

        # Fill std NaN (customers with 1 transaction)
        temporal_agg["hour_std"] = temporal_agg["hour_std"].fillna(0)

        logger.info(f"Temporal features engineered: {temporal_agg.shape[1]} columns")
        return temporal_agg


# =============================================================================
# 3. CATEGORICAL AGGREGATOR (Task 3)
# =============================================================================

class CategoricalAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregate categorical features to customer level.
    Uses dominant category + entropy/diversity measures.
    """

    def __init__(
        self,
        customer_id_col: str = "CustomerId",
        cat_cols: Optional[List[str]] = None,
    ):
        self.customer_id_col = customer_id_col
        self.cat_cols = cat_cols or [
            "ProductCategory",
            "ChannelId",
            "PricingStrategy",
            "ProviderId",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Aggregating categorical features per customer...")

        df = X.copy()
        result = df[[self.customer_id_col]].drop_duplicates()

        for col in self.cat_cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping.")
                continue

            # Dominant category per customer
            mode_df = (
                df.groupby(self.customer_id_col)[col]
                .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
                .reset_index(name=f"{col}_dominant")
            )

            # Category diversity (number of unique categories)
            nunique_df = (
                df.groupby(self.customer_id_col)[col]
                .nunique()
                .reset_index(name=f"{col}_diversity")
            )

            result = result.merge(mode_df, on=self.customer_id_col, how="left")
            result = result.merge(nunique_df, on=self.customer_id_col, how="left")

        logger.info(f"Categorical aggregation complete: {result.shape[1]} columns")
        return result


# =============================================================================
# 4. RFM CALCULATOR (Task 4)
# =============================================================================

class RFMCalculator(BaseEstimator, TransformerMixin):
    """
    Calculate Recency, Frequency, and Monetary values per customer.
    """

    def __init__(
        self,
        customer_id_col: str = "CustomerId",
        amount_col: str = "Amount",
        value_col: str = "Value",
        timestamp_col: str = "TransactionStartTime",
        snapshot_date: Optional[str] = None,
    ):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.value_col = value_col
        self.timestamp_col = timestamp_col
        self.snapshot_date = snapshot_date

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating RFM metrics...")

        df = X.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        if self.snapshot_date is None:
            self.snapshot_date = df[self.timestamp_col].max() + pd.Timedelta(days=1)
        else:
            self.snapshot_date = pd.to_datetime(self.snapshot_date)

        rfm = (
            df.groupby(self.customer_id_col)
            .agg(
                recency=(self.timestamp_col, lambda x: (self.snapshot_date - x.max()).days),
                frequency=(self.amount_col, "count"),
                monetary=(self.value_col, "sum"),
            )
            .reset_index()
        )

        # Use absolute monetary and log-transform (EDA: skewness > 50)
        rfm["monetary"] = rfm["monetary"].abs()
        rfm["log_monetary"] = np.log1p(rfm["monetary"])

        logger.info(f"RFM calculated for {len(rfm)} customers")
        return rfm


# =============================================================================
# 5. RISK LABEL ASSIGNER (Task 4)
# =============================================================================

class RiskLabelAssigner(BaseEstimator, TransformerMixin):
    """
    Assign is_high_risk labels using K-Means clustering on RFM features.
    High-risk = high recency, low frequency, low monetary.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_ = None
        self.scaler_ = None
        self.high_risk_cluster_ = None

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting K-Means for risk segmentation...")

        rfm_cols = ["recency", "frequency", "log_monetary"]
        if not all(c in X.columns for c in rfm_cols):
            raise ValueError(f"RFM columns {rfm_cols} must be present in input DataFrame")

        # Scale RFM for clustering
        self.scaler_ = StandardScaler()
        rfm_scaled = self.scaler_.fit_transform(X[rfm_cols])

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans_.fit(rfm_scaled)

        # Identify high-risk cluster: high recency + low frequency + low monetary
        centers = pd.DataFrame(
            self.kmeans_.cluster_centers_,
            columns=rfm_cols,
        )

        # Composite risk score: higher recency is bad, lower frequency is bad, lower monetary is bad
        centers["risk_score"] = (
            centers["recency"]
            - centers["frequency"]
            - centers["log_monetary"]
        )
        self.high_risk_cluster_ = int(centers["risk_score"].idxmax())

        logger.info(f"High-risk cluster identified: Cluster {self.high_risk_cluster_}")
        logger.info(f"Cluster centers:\n{centers.round(3)}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Assigning risk labels...")

        rfm_cols = ["recency", "frequency", "log_monetary"]
        rfm_scaled = self.scaler_.transform(X[rfm_cols])

        clusters = self.kmeans_.predict(rfm_scaled)
        X = X.copy()
        X["cluster"] = clusters
        X["is_high_risk"] = (clusters == self.high_risk_cluster_).astype(int)

        risk_dist = X["is_high_risk"].value_counts()
        logger.info(f"Risk distribution: {risk_dist.to_dict()}")

        return X


# =============================================================================
# 6. WoE / IV TRANSFORMER (Task 3)
# =============================================================================

class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Manual Weight-of-Evidence (WoE) and Information Value (IV) transformer.
    Falls back to quantile-based binning if xverse is not installed.
    """

    def __init__(
        self,
        columns: List[str],
        target_col: str = "is_high_risk",
        n_bins: int = 10,
        use_xverse: bool = False,
    ):
        self.columns = columns
        self.target_col = target_col
        self.n_bins = n_bins
        self.use_xverse = use_xverse
        self.woe_maps_: Dict[str, Dict] = {}
        self.iv_values_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None):
        if self.target_col not in X.columns:
            raise ValueError(f"Target column '{self.target_col}' not found")

        df = X.copy()
        y = df[self.target_col]

        for col in self.columns:
            if col not in df.columns:
                continue

            # Quantile-based binning (monotonic, handles outliers)
            try:
                bins = pd.qcut(df[col], q=self.n_bins, duplicates="drop")
            except ValueError:
                bins = pd.cut(df[col], bins=self.n_bins)

            # Calculate WoE per bin
            woe_df = pd.DataFrame({"bin": bins, "target": y})
            grouped = woe_df.groupby("bin", observed=False)["target"].agg(["count", "sum"])
            grouped["non_event"] = grouped["count"] - grouped["sum"]
            grouped["event_rate"] = grouped["sum"] / grouped["sum"].sum()
            grouped["non_event_rate"] = grouped["non_event"] / grouped["non_event"].sum()

            # Smoothing to avoid division by zero
            grouped["event_rate"] = grouped["event_rate"].replace(0, 0.0001)
            grouped["non_event_rate"] = grouped["non_event_rate"].replace(0, 0.0001)

            grouped["woe"] = np.log(grouped["event_rate"] / grouped["non_event_rate"])
            grouped["iv"] = (grouped["event_rate"] - grouped["non_event_rate"]) * grouped["woe"]

            self.woe_maps_[col] = grouped["woe"].to_dict()
            self.iv_values_[col] = grouped["iv"].sum()

        logger.info(f"WoE fitted for columns: {list(self.woe_maps_.keys())}")
        logger.info(f"IV values: { {k: round(v, 3) for k, v in self.iv_values_.items()} }")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.columns:
            if col not in X.columns or col not in self.woe_maps_:
                continue

            # Re-bin using same cut points, then map WoE
            try:
                bins = pd.qcut(X[col], q=self.n_bins, duplicates="drop")
            except ValueError:
                bins = pd.cut(X[col], bins=self.n_bins)

            woe_map = self.woe_maps_[col]
            # Map each bin to its WoE value
            X[f"{col}_woe"] = bins.map(woe_map).astype(float)

        return X


# =============================================================================
# 7. DATA CLEANER / DEFENSIVE PREPROCESSOR (Task 3)
# =============================================================================

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Defensive cleaning based on EDA findings:
    - Drop constant columns (e.g., CountryCode)
    - Handle infinities
    - Ensure correct dtypes
    """

    def __init__(self, drop_constant: bool = True):
        self.drop_constant = drop_constant
        self.constant_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if self.drop_constant:
            self.constant_cols_ = [
                col for col in X.columns
                if X[col].nunique(dropna=False) <= 1
            ]
            logger.info(f"Constant columns to drop: {self.constant_cols_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.constant_cols_:
            X = X.drop(columns=self.constant_cols_, errors="ignore")

        # Replace infinities
        X = X.replace([np.inf, -np.inf], np.nan)

        logger.info(f"Data cleaned: {X.shape[1]} columns remaining")
        return X


# =============================================================================
# 8. FULL PIPELINE BUILDER
# =============================================================================

def build_processing_pipeline(
    snapshot_date: Optional[str] = None,
    n_clusters: int = 3,
) -> Pipeline:
    """
    Build the complete sklearn Pipeline for Tasks 3 & 4.
    
    Note: This pipeline handles the full flow from raw transactions to
    customer-level model-ready data. Because aggregation changes the
    DataFrame shape, we use a custom orchestration function rather than
    a pure sklearn Pipeline for the aggregation step.
    """
    # The aggregation and RFM steps are done in process_data() below.
    # This pipeline handles the post-aggregation preprocessing.
    pass


def process_data(
    raw_df: pd.DataFrame,
    snapshot_date: Optional[str] = None,
    n_clusters: int = 3,
    apply_woe: bool = True,
) -> pd.DataFrame:
    """
    Main orchestration function.
    
    Takes raw transaction-level DataFrame and returns customer-level,
    model-ready DataFrame with is_high_risk target and engineered features.
    
    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw Xente transaction data
    snapshot_date : str, optional
        Snapshot date for recency calculation. Defaults to max date + 1 day.
    n_clusters : int
        Number of K-Means clusters for risk segmentation
    apply_woe : bool
        Whether to apply WoE transformation
    
    Returns
    -------
    pd.DataFrame
        Processed customer-level dataset ready for modeling
    """
    logger.info(f"Starting data processing pipeline on {raw_df.shape[0]:,} transactions")

    # Step 1: Aggregate numerical features
    agg_engineer = AggregateFeatureEngineer()
    agg_features = agg_engineer.fit_transform(raw_df)

    # Step 2: Temporal features
    temporal_engineer = TemporalFeatureEngineer()
    temporal_features = temporal_engineer.fit_transform(raw_df)

    # Step 3: Categorical aggregation
    cat_aggregator = CategoricalAggregator()
    cat_features = cat_aggregator.fit_transform(raw_df)

    # Step 4: Merge all customer-level features
    customer_df = agg_features.merge(
        temporal_features, on="CustomerId", how="outer"
    ).merge(
        cat_features, on="CustomerId", how="outer"
    )

    # Step 5: RFM calculation
    rfm_calc = RFMCalculator(snapshot_date=snapshot_date)
    rfm_features = rfm_calc.fit_transform(raw_df)

    # Step 6: Merge RFM into customer dataset
    customer_df = customer_df.merge(rfm_features, on="CustomerId", how="outer")

    # Step 7: Data cleaning (drop constant columns, handle inf/nan)
    cleaner = DataCleaner()
    customer_df = cleaner.fit_transform(customer_df)

    # Step 8: Defensive imputation for any remaining NaNs
    # (EDA showed no missing values, but pipeline must be robust)
    num_cols = customer_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "CustomerId"]
    
    for col in num_cols:
        if customer_df[col].isnull().any():
            customer_df[col] = customer_df[col].fillna(customer_df[col].median())

    # Step 9: Risk label assignment via K-Means (Task 4)
    risk_assigner = RiskLabelAssigner(n_clusters=n_clusters, random_state=RANDOM_STATE)
    customer_df = risk_assigner.fit_transform(customer_df)

    # Step 10: WoE transformation (Task 3)
    if apply_woe:
        # Select numerical features for WoE (exclude ID, target, cluster)
        woe_candidates = [
            c for c in num_cols
            if c not in ["recency", "frequency", "monetary", "log_monetary", "cluster", "is_high_risk"]
            and customer_df[c].nunique() > 5  # Need enough variation to bin
        ]
        
        if woe_candidates:
            woe_transformer = WoETransformer(
                columns=woe_candidates[:5],  # Limit to top 5 to avoid over-engineering
                target_col="is_high_risk",
                n_bins=10,
            )
            customer_df = woe_transformer.fit_transform(customer_df)
            logger.info(f"WoE applied to columns: {woe_candidates[:5]}")

    # Step 11: Final feature selection / ordering
    # Drop raw high-cardinality IDs that are not model features
    drop_cols = ["CustomerId"]  # Keep if needed for reference, but exclude from modeling
    # Actually keep CustomerId for reference, drop only if explicitly requested

    logger.info(f"Pipeline complete. Output shape: {customer_df.shape}")
    logger.info(f"Final columns: {list(customer_df.columns)}")

    return customer_df


# =============================================================================
# 9. UTILITY FUNCTIONS
# =============================================================================

def get_feature_columns(df: pd.DataFrame, target_col: str = "is_high_risk") -> List[str]:
    """
    Return list of feature columns suitable for model training.
    Excludes target, cluster label, and ID columns.
    """
    exclude = {target_col, "cluster", "CustomerId"}
    return [c for c in df.columns if c not in exclude]


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "is_high_risk",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split processed DataFrame into X (features) and y (target).
    """
    features = get_feature_columns(df, target_col)
    X = df[features]
    y = df[target_col]
    return X, y


# =============================================================================
# 10. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example usage (requires data/raw/training.csv)
    import os

    data_path = os.path.join("data", "raw", "training.csv")
    if os.path.exists(data_path):
        raw = pd.read_csv(data_path)
        processed = process_data(raw)
        processed.to_csv("data/processed/customer_features.csv", index=False)
        logger.info("Processed data saved to data/processed/customer_features.csv")
    else:
        logger.warning(f"Data file not found at {data_path}. Run this after placing the dataset.")