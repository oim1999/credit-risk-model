"""
Credit Risk Feature Engineering Pipeline
========================================
Transforms raw Xente transaction data into model-ready features.
Covers Tasks 3 (Feature Engineering) and Task 4 (Proxy Target Variable).

Usage:
    from src.data_processing import FeatureEngineeringPipeline, RFMCalculator
    
    pipeline = FeatureEngineeringPipeline()
    X_processed, y_proxy = pipeline.fit_transform(df_raw)
"""

import logging
import warnings
from datetime import datetime
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder

# Optional: xverse for WoE/IV
# Install: pip install xverse
# For this pipeline, we implement a custom WoE transformer if xverse is unavailable
try:
    from xverse.transformer import WOE
    HAS_XVERSE = True
except ImportError:
    HAS_XVERSE = False
    warnings.warn("xverse not installed. Using custom WoE implementation. "
                  "Install with: pip install xverse")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
SNAPSHOT_DATE = pd.Timestamp("2019-01-01")  # Adjust based on your data's max date

# Columns expected in raw data
RAW_COLS_EXPECTED = [
    "TransactionId", "BatchId", "AccountId", "SubscriptionId",
    "CustomerId", "CurrencyCode", "CountryCode", "ProviderId",
    "ProductId", "ProductCategory", "ChannelId", "Amount",
    "Value", "TransactionStartTime", "PricingStrategy", "FraudResult"
]

CATEGORICAL_COLS = ["ProductCategory", "ChannelId", "PricingStrategy"]
NUMERICAL_COLS = ["Amount", "Value"]  # Before aggregation
TEMPORAL_COLS = ["TransactionStartTime"]

# Rare category threshold: group categories with < 1% frequency into "Other"
RARE_CATEGORY_THRESHOLD = 0.01


# =============================================================================
# CUSTOM TRANSFORMERS
# =============================================================================

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Group rare categorical values into a single 'Other' category.
    Reduces dimensionality and avoids sparse one-hot features.
    """

    def __init__(self, threshold: float = RARE_CATEGORY_THRESHOLD):
        self.threshold = threshold
        self.freq_maps_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in X.select_dtypes(include=["object", "category"]).columns:
            freq = X[col].value_counts(normalize=True)
            self.freq_maps_[col] = set(freq[freq >= self.threshold].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, valid_cats in self.freq_maps_.items():
            if col in X.columns:
                X[col] = X[col].apply(
                    lambda v: v if v in valid_cats else "Other"
                )
        return X


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract hour, day, month, year, weekday, weekend_flag from datetime.
    Also adds cyclical (sin/cos) encoding for hour and month.
    """

    def __init__(self, datetime_col: str = "TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.datetime_col not in X.columns:
            raise ValueError(f"Column '{self.datetime_col}' not found in DataFrame.")

        dt = pd.to_datetime(X[self.datetime_col], errors="coerce")

        X["txn_hour"] = dt.dt.hour
        X["txn_day"] = dt.dt.day
        X["txn_month"] = dt.dt.month
        X["txn_year"] = dt.dt.year
        X["txn_weekday"] = dt.dt.weekday  # 0=Monday
        X["txn_is_weekend"] = (dt.dt.weekday >= 5).astype(int)

        # Cyclical encoding: captures that 23:00 is close to 00:00
        X["txn_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        X["txn_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        X["txn_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        X["txn_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)

        # Drop raw datetime column (kept only if needed downstream)
        X = X.drop(columns=[self.datetime_col])
        return X


class AggregateFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Compute per-customer aggregate features from transaction-level data.
    Output: one row per CustomerId with aggregate statistics.
    """

    def __init__(self, customer_id_col: str = "CustomerId",
                 amount_col: str = "Amount",
                 value_col: str = "Value"):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.value_col = value_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.customer_id_col not in X.columns:
            raise ValueError(f"Customer ID column '{self.customer_id_col}' not found.")

        grp = X.groupby(self.customer_id_col)

        agg_specs = {
            # Amount-based features
            f"{self.amount_col}_sum": pd.NamedAgg(column=self.amount_col, aggfunc="sum"),
            f"{self.amount_col}_mean": pd.NamedAgg(column=self.amount_col, aggfunc="mean"),
            f"{self.amount_col}_std": pd.NamedAgg(column=self.amount_col, aggfunc="std"),
            f"{self.amount_col}_max": pd.NamedAgg(column=self.amount_col, aggfunc="max"),
            f"{self.amount_col}_min": pd.NamedAgg(column=self.amount_col, aggfunc="min"),
            f"{self.amount_col}_count": pd.NamedAgg(column=self.amount_col, aggfunc="count"),
            f"{self.amount_col}_median": pd.NamedAgg(column=self.amount_col, aggfunc="median"),

            # Value-based features (absolute magnitude)
            f"{self.value_col}_sum": pd.NamedAgg(column=self.value_col, aggfunc="sum"),
            f"{self.value_col}_mean": pd.NamedAgg(column=self.value_col, aggfunc="mean"),
            f"{self.value_col}_std": pd.NamedAgg(column=self.value_col, aggfunc="std"),
            f"{self.value_col}_max": pd.NamedAgg(column=self.value_col, aggfunc="max"),
            f"{self.value_col}_median": pd.NamedAgg(column=self.value_col, aggfunc="median"),

            # Fraud-related
            "fraud_count": pd.NamedAgg(column="FraudResult", aggfunc="sum"),
            "fraud_rate": pd.NamedAgg(column="FraudResult", aggfunc="mean"),

            # Temporal diversity
            "txn_hour_std": pd.NamedAgg(column="txn_hour", aggfunc="std"),
            "unique_products": pd.NamedAgg(column="ProductCategory", aggfunc="nunique"),
            "unique_channels": pd.NamedAgg(column="ChannelId", aggfunc="nunique"),
        }

        customer_df = grp.agg(**agg_specs).reset_index()

        # Fill NaN stds (customers with only 1 transaction get std=0)
        std_cols = [c for c in customer_df.columns if c.endswith("_std")]
        customer_df[std_cols] = customer_df[std_cols].fillna(0)

        # Log-transform monetary aggregates to tame extreme skewness (>50)
        for col in customer_df.columns:
            if any(suffix in col for suffix in ["_sum", "_mean", "_std", "_max", "_median"]):
                if customer_df[col].min() >= 0:
                    customer_df[f"{col}_log1p"] = np.log1p(customer_df[col])

        logger.info(f"Aggregate features computed for {len(customer_df)} customers.")
        return customer_df


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence binning for numerical features.
    Used primarily for Logistic Regression baseline to create
    monotonic, interpretable, regulator-friendly features.

    If xverse is available, uses WOE from xverse.transformer.
    Otherwise, falls back to a manual pandas-based implementation.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.woe_maps_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if y is None:
            raise ValueError("WoE transformer requires target variable y during fit.")

        for col in X.select_dtypes(include=[np.number]).columns:
            try:
                if HAS_XVERSE:
                    woe = WOE()
                    woe.fit(X[[col]], y)
                    # Store the transformer for later use
                    self.woe_maps_[col] = woe
                else:
                    self.woe_maps_[col] = self._fit_manual_woe(X[col], y)
            except Exception as e:
                logger.warning(f"WoE fit failed for column '{col}': {e}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_woe = X.copy()
        for col, woe_obj in self.woe_maps_.items():
            if col not in X_woe.columns:
                continue
            try:
                if HAS_XVERSE:
                    X_woe[f"{col}_woe"] = woe_obj.transform(X_woe[[col]])
                else:
                    X_woe[f"{col}_woe"] = X_woe[col].map(woe_obj)
                    # Fill unseen values with 0 (neutral WoE)
                    X_woe[f"{col}_woe"] = X_woe[f"{col}_woe"].fillna(0)
            except Exception as e:
                logger.warning(f"WoE transform failed for column '{col}': {e}")
        return X_woe

    def _fit_manual_woe(self, x: pd.Series, y: pd.Series) -> dict:
        """Manual WoE calculation using quantile-based bins."""
        # Create bins using quantiles
        bins = pd.qcut(x, q=self.n_bins, duplicates="drop")
        woe_df = pd.DataFrame({"bin": bins, "target": y})

        # Calculate distribution of goods (y=0) and bads (y=1) per bin
        grouped = woe_df.groupby("bin")["target"].agg(["sum", "count"])
        grouped["bad"] = grouped["sum"]
        grouped["good"] = grouped["count"] - grouped["sum"]

        # Add small constant to avoid division by zero
        eps = 0.5
        total_good = (y == 0).sum() + eps
        total_bad = (y == 1).sum() + eps

        grouped["woe"] = np.log(
            (grouped["good"] + eps) / total_good /
            ((grouped["bad"] + eps) / total_bad)
        )

        # Return mapping from bin interval to WoE value
        return grouped["woe"].to_dict()


# =============================================================================
# RFM CALCULATOR (TASK 4)
# =============================================================================

class RFMCalculator(BaseEstimator, TransformerMixin):
    """
    Calculate Recency, Frequency, and Monetary (RFM) values per customer.
    
    Recency: Days since last transaction (relative to SNAPSHOT_DATE).
    Frequency: Number of transactions.
    Monetary: Total transaction value (absolute).
    
    These RFM features feed into K-Means clustering to create the proxy target.
    """

    def __init__(self,
                 customer_id_col: str = "CustomerId",
                 value_col: str = "Value",
                 datetime_col: str = "TransactionStartTime",
                 snapshot_date: Optional[pd.Timestamp] = None):
        self.customer_id_col = customer_id_col
        self.value_col = value_col
        self.datetime_col = datetime_col
        self.snapshot_date = snapshot_date or SNAPSHOT_DATE

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dt = pd.to_datetime(X[self.datetime_col], errors="coerce")

        rfm = X.groupby(self.customer_id_col).agg(
            recency_days=(self.datetime_col, lambda ts: (
                self.snapshot_date - pd.to_datetime(ts).max()
            ).days),
            frequency=(self.value_col, "count"),
            monetary=(self.value_col, "sum")
        ).reset_index()

        # Log-transform monetary to handle skewness
        rfm["monetary_log"] = np.log1p(rfm["monetary"])

        logger.info(f"RFM calculated for {len(rfm)} customers. "
                    f"Snapshot date: {self.snapshot_date.date()}")
        return rfm


class RiskProxyAssigner(BaseEstimator, TransformerMixin):
    """
    Assign is_high_risk proxy target using K-Means clustering on RFM features.

    High-risk cluster = lowest frequency + lowest monetary + highest recency
    (i.e., disengaged, low-value, dormant customers).
    
    The cluster with the lowest combined engagement score is labeled as high-risk.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_ = None
        self.high_risk_cluster_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        # X expected to have: recency_days, frequency, monetary_log
        rfm_features = X[["recency_days", "frequency", "monetary_log"]].copy()
        rfm_scaled = self.scaler_.fit_transform(rfm_features)

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        clusters = self.kmeans_.fit_predict(rfm_scaled)

        # Identify high-risk cluster: lowest frequency + lowest monetary + highest recency
        cluster_profiles = pd.DataFrame({
            "cluster": range(self.n_clusters),
            "mean_recency": self.kmeans_.cluster_centers_[:, 0],
            "mean_frequency": self.kmeans_.cluster_centers_[:, 1],
            "mean_monetary": self.kmeans_.cluster_centers_[:, 2]
        })

        # Normalize profiles to 0-1 for scoring
        for col in ["mean_recency", "mean_frequency", "mean_monetary"]:
            mn, mx = cluster_profiles[col].min(), cluster_profiles[col].max()
            if mx > mn:
                cluster_profiles[col] = (cluster_profiles[col] - mn) / (mx - mn)

        # Risk score: high recency is bad, low frequency is bad, low monetary is bad
        cluster_profiles["risk_score"] = (
            cluster_profiles["mean_recency"] * 1.0      # higher recency = riskier
            + (1 - cluster_profiles["mean_frequency"]) * 1.0  # lower freq = riskier
            + (1 - cluster_profiles["mean_monetary"]) * 0.5   # lower monetary = riskier
        )

        self.high_risk_cluster_ = int(cluster_profiles["risk_score"].idxmax())

        logger.info(
            f"K-Means fitted with {self.n_clusters} clusters. "
            f"High-risk cluster identified: Cluster {self.high_risk_cluster_}\n"
            f"Cluster profiles:\n{cluster_profiles}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        rfm_features = X[["recency_days", "frequency", "monetary_log"]].copy()
        rfm_scaled = self.scaler_.transform(rfm_features)
        clusters = self.kmeans_.predict(rfm_scaled)

        X = X.copy()
        X["cluster"] = clusters
        X["is_high_risk"] = (clusters == self.high_risk_cluster_).astype(int)

        risk_rate = X["is_high_risk"].mean()
        logger.info(f"Proxy target assigned. High-risk rate: {risk_rate:.2%}")
        return X


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class FeatureEngineeringPipeline:
    """
    End-to-end pipeline that transforms raw Xente transaction data
    into model-ready features with a proxy target variable.

    Steps:
        1. Validate raw data columns
        2. Extract temporal features
        3. Group rare categories
        4. Compute per-customer aggregates
        5. Encode categorical variables (one-hot)
        6. Scale numerical features
        7. Calculate RFM features
        8. Assign proxy target via K-Means clustering
        9. (Optional) Apply WoE transformation

    Parameters
    ----------
    apply_woe : bool
        If True, applies Weight-of-Evidence transformation to numerical features.
        Recommended for Logistic Regression baseline.
    woe_target_col : str
        Column name to use as target for WoE calculation (typically 'is_high_risk').
    """

    def __init__(self,
                 apply_woe: bool = False,
                 woe_target_col: str = "is_high_risk",
                 snapshot_date: Optional[pd.Timestamp] = None):
        self.apply_woe = apply_woe
        self.woe_target_col = woe_target_col
        self.snapshot_date = snapshot_date or SNAPSHOT_DATE

        # Sub-components (initialized during fit)
        self.rare_grouper_ = RareCategoryGrouper(threshold=RARE_CATEGORY_THRESHOLD)
        self.temporal_extractor_ = TemporalFeatureExtractor()
        self.agg_engineer_ = AggregateFeatureEngineer()
        self.rfm_calculator_ = RFMCalculator(snapshot_date=self.snapshot_date)
        self.risk_assigner_ = RiskProxyAssigner()
        self.onehot_encoder_ = None
        self.scaler_ = StandardScaler()
        self.woe_transformer_ = WoETransformer(n_bins=10) if apply_woe else None

        # Column tracking
        self.feature_names_: Optional[List[str]] = None
        self.target_col_ = "is_high_risk"

    def fit_transform(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the pipeline on raw data and return processed features + proxy target.

        Returns
        -------
        X : pd.DataFrame
            Model-ready feature matrix (one row per customer).
        y : pd.Series
            Proxy target variable (is_high_risk).
        """
        logger.info("=" * 60)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 60)

        # ---- Step 0: Validation ----
        self._validate_raw_data(df_raw)
        df = df_raw.copy()
        logger.info(f"Raw data shape: {df.shape}")

        # ---- Step 1: Temporal Feature Extraction ----
        logger.info("[Step 1/8] Extracting temporal features...")
        df = self.temporal_extractor_.fit_transform(df)

        # ---- Step 2: Group Rare Categories ----
        logger.info("[Step 2/8] Grouping rare categorical categories...")
        df = self.rare_grouper_.fit_transform(df)

        # ---- Step 3: Aggregate Features per Customer ----
        logger.info("[Step 3/8] Computing per-customer aggregate features...")
        customer_agg = self.agg_engineer_.fit_transform(df)

        # ---- Step 4: RFM Calculation ----
        logger.info("[Step 4/8] Calculating RFM features...")
        # Need CustomerId + datetime + Value from original df for RFM
        rfm_df = self.rfm_calculator_.fit_transform(df_raw)

        # ---- Step 5: Proxy Target Assignment (K-Means) ----
        logger.info("[Step 5/8] Assigning proxy target via K-Means clustering...")
        rfm_with_target = self.risk_assigner_.fit_transform(rfm_df)

        # Merge RFM + target into customer aggregates
        customer_id_col = self.agg_engineer_.customer_id_col
        customer_agg = customer_agg.merge(
            rfm_with_target[[customer_id_col, "recency_days", "frequency",
                            "monetary", "monetary_log", "cluster", "is_high_risk"]],
            on=customer_id_col,
            how="left"
        )

        # Separate target before encoding/scaling
        y = customer_agg[self.target_col_].copy()
        customer_agg = customer_agg.drop(columns=[self.target_col_])

        # ---- Step 6: Categorical Encoding ----
        logger.info("[Step 6/8] Encoding categorical variables...")
        cat_cols_present = [c for c in CATEGORICAL_COLS if c in customer_agg.columns]

        if cat_cols_present:
            # Use pandas get_dummies for simplicity and feature name preservation
            customer_agg = pd.get_dummies(customer_agg, columns=cat_cols_present,
                                          drop_first=False)

        # ---- Step 7: Scaling ----
        logger.info("[Step 7/8] Scaling numerical features...")
        num_cols = customer_agg.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude identifier columns from scaling
        exclude_from_scaling = [customer_id_col, "cluster"]
        scale_cols = [c for c in num_cols if c not in exclude_from_scaling]

        if scale_cols:
            customer_agg[scale_cols] = self.scaler_.fit_transform(
                customer_agg[scale_cols]
            )

        # ---- Step 8: WoE Transformation (Optional) ----
        if self.apply_woe and self.woe_transformer_ is not None:
            logger.info("[Step 8/8] Applying WoE transformation...")
            # Temporarily attach target for WoE fitting
            customer_agg[self.woe_target_col] = y.values
            customer_agg = self.woe_transformer_.fit_transform(
                customer_agg, y
            )
            customer_agg = customer_agg.drop(columns=[self.woe_target_col])

        self.feature_names_ = [c for c in customer_agg.columns
                               if c != customer_id_col]

        logger.info(f"Pipeline complete. Final feature matrix: {customer_agg[self.feature_names_].shape}")
        logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")

        return customer_agg, y

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new raw data using fitted pipeline.
        NOTE: This method assumes the proxy target is NOT available.
        It produces features only (for inference on new customers).
        """
        df = df_raw.copy()
        df = self.temporal_extractor_.transform(df)
        df = self.rare_grouper_.transform(df)
        customer_agg = self.agg_engineer_.transform(df)

        # For inference, we skip RFM-based target assignment
        # (new customers won't have a target yet)
        rfm_df = self.rfm_calculator_.transform(df_raw)
        rfm_features = rfm_df[["recency_days", "frequency", "monetary", "monetary_log"]]

        customer_id_col = self.agg_engineer_.customer_id_col
        customer_agg = customer_agg.merge(
            rfm_df[[customer_id_col, "recency_days", "frequency",
                    "monetary", "monetary_log"]],
            on=customer_id_col,
            how="left"
        )

        cat_cols_present = [c for c in CATEGORICAL_COLS if c in customer_agg.columns]
        if cat_cols_present:
            customer_agg = pd.get_dummies(customer_agg, columns=cat_cols_present,
                                          drop_first=False)

        num_cols = customer_agg.select_dtypes(include=[np.number]).columns.tolist()
        exclude_from_scaling = [customer_id_col]
        scale_cols = [c for c in num_cols if c not in exclude_from_scaling]

        if scale_cols:
            customer_agg[scale_cols] = self.scaler_.transform(customer_agg[scale_cols])

        # Ensure column alignment with training-time features
        if self.feature_names_:
            for col in self.feature_names_:
                if col not in customer_agg.columns:
                    customer_agg[col] = 0
            customer_agg = customer_agg[[customer_id_col] + self.feature_names_]

        return customer_agg

    def _validate_raw_data(self, df: pd.DataFrame):
        """Ensure required columns are present."""
        missing = [c for c in RAW_COLS_EXPECTED if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        logger.info("Raw data validation passed.")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """
    CLI entry point for the feature engineering pipeline.
    
    Usage:
        python src/data_processing.py --input data/raw/training.csv --output data/processed/
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument("--input", required=True, help="Path to raw CSV data")
    parser.add_argument("--output", required=True, help="Output directory for processed data")
    parser.add_argument("--apply-woe", action="store_true", help="Apply WoE transformation")
    parser.add_argument("--snapshot-date", default="2019-01-01",
                        help="Snapshot date for RFM recency calculation (YYYY-MM-DD)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logger.info(f"Loading raw data from {args.input}")
    df_raw = pd.read_csv(args.input)

    snapshot = pd.Timestamp(args.snapshot_date)
    pipeline = FeatureEngineeringPipeline(apply_woe=args.apply_woe,
                                          snapshot_date=snapshot)
    X, y = pipeline.fit_transform(df_raw)

    # Save outputs
    output_path = os.path.join(args.output, "train_processed.csv")
    X[self.target_col_] = y.values
    X.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")

    # Save feature list for API reference
    feature_list_path = os.path.join(args.output, "feature_names.txt")
    with open(feature_list_path, "w") as f:
        f.write("\n".join(pipeline.feature_names_))
    logger.info(f"Saved feature names to {feature_list_path}")


if __name__ == "__main__":
    main()