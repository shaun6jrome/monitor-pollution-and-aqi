"""Feature Engineering and Chronological Data Splitting for Air Quality Forecasting.

Constructs diurnal/seasonal cyclical encodings, historical lag variables, rolling window
aggregations, and aligned predictor matrices. Implements strict leakage-free imputation
where transformation parameters are computed solely on training partitions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

DEFAULT_PREDICTOR_SENSORS = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cyclical hour and month features alongside calendar indicators."""
    res = df.copy()
    dt = res["datetime"].dt

    res["hour"] = dt.hour
    res["day_of_week"] = dt.dayofweek
    res["month"] = dt.month
    res["is_weekend"] = (dt.dayofweek >= 5).astype(int)

    # Cyclical hour encoding (24-hour cycle)
    res["hour_sin"] = np.sin(2 * np.pi * res["hour"] / 24.0)
    res["hour_cos"] = np.cos(2 * np.pi * res["hour"] / 24.0)

    # Cyclical month encoding (12-month cycle)
    res["month_sin"] = np.sin(2 * np.pi * res["month"] / 12.0)
    res["month_cos"] = np.cos(2 * np.pi * res["month"] / 12.0)

    return res


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = config.DEFAULT_FORECAST_TARGET,
) -> pd.DataFrame:
    """Create strictly causal past lag variables and rolling statistics without forward lookahead."""
    res = df.copy()

    # Past lags of target variable
    res[f"{target_col}_lag_1"] = res[target_col].shift(1)
    res[f"{target_col}_lag_2"] = res[target_col].shift(2)
    res[f"{target_col}_lag_3"] = res[target_col].shift(3)
    res[f"{target_col}_lag_24"] = res[target_col].shift(24)

    # Rolling statistics of target over past observations (closed='left' avoids lookahead)
    rolling_obj = res[target_col].shift(1).rolling(window=6, min_periods=2)
    res[f"{target_col}_rolling_mean_6h"] = rolling_obj.mean()
    res[f"{target_col}_rolling_std_6h"] = rolling_obj.std()

    # Sensor lags
    for col in ["PT08.S1(CO)", "T", "RH"]:
        if col in res.columns:
            res[f"{col}_lag_1"] = res[col].shift(1)

    return res


def prepare_forecasting_dataset(
    df: pd.DataFrame,
    target_col: str = config.DEFAULT_FORECAST_TARGET,
    test_ratio: float = 0.20,
) -> Dict[str, Any]:
    """Build feature matrix and partition into chronological train and test sets.

    Guarantees no temporal leakage across the split boundary.
    """
    df_sorted = df.sort_values("datetime").reset_index(drop=True)

    # 1. Generate temporal and lag features
    with_time = create_time_features(df_sorted)
    with_lags = create_lag_and_rolling_features(with_time, target_col=target_col)

    # Feature column selection
    feature_cols = [
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        f"{target_col}_lag_1",
        f"{target_col}_lag_2",
        f"{target_col}_lag_3",
        f"{target_col}_lag_24",
        f"{target_col}_rolling_mean_6h",
        f"{target_col}_rolling_std_6h",
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH",
        "PT08.S1(CO)_lag_1",
        "T_lag_1",
        "RH_lag_1",
    ]

    # Filter to columns that actually exist in dataframe
    active_features = [c for c in feature_cols if c in with_lags.columns]

    # Filter out rows where the target is missing
    valid_mask = with_lags[target_col].notna()
    clean_data = with_lags[valid_mask].copy().reset_index(drop=True)

    # Chronological train/test split boundary
    n_total = len(clean_data)
    split_idx = int(n_total * (1.0 - test_ratio))

    train_df = clean_data.iloc[:split_idx].copy()
    test_df = clean_data.iloc[split_idx:].copy()

    # Leakage-free imputation: compute median values strictly on training features
    train_medians = train_df[active_features].median().to_dict()

    x_train = train_df[active_features].fillna(train_medians).values
    y_train = train_df[target_col].values

    x_test = test_df[active_features].fillna(train_medians).values
    y_test = test_df[target_col].values

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "train_timestamps": train_df["datetime"].tolist(),
        "test_timestamps": test_df["datetime"].tolist(),
        "train_period": (train_df["datetime"].min().isoformat(), train_df["datetime"].max().isoformat()),
        "test_period": (test_df["datetime"].min().isoformat(), test_df["datetime"].max().isoformat()),
        "feature_names": active_features,
        "train_medians": train_medians,
        "target_name": target_col,
        "target_unit": "mg/m^3" if "CO" in target_col else "microg/m^3",
    }
