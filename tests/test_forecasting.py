"""Unit tests for ML forecasting pipeline, feature engineering, and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.evaluate_models import compute_residual_diagnostics
from src.forecasting.feature_engineering import create_time_features, prepare_forecasting_dataset
from src.forecasting.predict import ModelPredictor


@pytest.fixture
def mock_air_quality_df():
    """Create synthetic historical dataframe matching UCI schema structure."""
    n_rows = 200
    dates = pd.date_range(start="2004-03-10 18:00:00", periods=n_rows, freq="h")
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "datetime": dates,
            "CO(GT)": np.random.uniform(0.5, 8.0, n_rows),
            "PT08.S1(CO)": np.random.uniform(700, 1800, n_rows),
            "PT08.S2(NMHC)": np.random.uniform(600, 1600, n_rows),
            "PT08.S3(NOx)": np.random.uniform(400, 1200, n_rows),
            "PT08.S4(NO2)": np.random.uniform(800, 2000, n_rows),
            "PT08.S5(O3)": np.random.uniform(500, 1800, n_rows),
            "T": np.random.uniform(5.0, 35.0, n_rows),
            "RH": np.random.uniform(20.0, 80.0, n_rows),
            "AH": np.random.uniform(0.5, 2.0, n_rows),
        }
    )
    # Inject some missing values simulating -200 sentinels
    df.loc[10:15, "CO(GT)"] = np.nan
    df.loc[30:35, "PT08.S1(CO)"] = np.nan
    return df


def test_time_feature_generation(mock_air_quality_df):
    """Verify calculation of cyclical sin/cos hour and month features."""
    df_feat = create_time_features(mock_air_quality_df)
    assert "hour_sin" in df_feat.columns
    assert "hour_cos" in df_feat.columns
    assert "is_weekend" in df_feat.columns
    # Sin/cos bounds
    assert df_feat["hour_sin"].between(-1.0, 1.0).all()
    assert df_feat["hour_cos"].between(-1.0, 1.0).all()


def test_chronological_splitting_no_leakage(mock_air_quality_df):
    """Ensure training and test sets are split strictly chronologically without temporal overlap."""
    pack = prepare_forecasting_dataset(mock_air_quality_df, target_col="CO(GT)", test_ratio=0.20)

    train_end = pack["train_period"][1]
    test_start = pack["test_period"][0]

    # Test start timestamp must be chronologically strictly greater than or equal to train end
    assert test_start >= train_end
    assert len(pack["x_train"]) > 0
    assert len(pack["x_test"]) > 0

    # Ensure no NaN values remain in feature matrices after imputation
    assert not np.isnan(pack["x_train"]).any()
    assert not np.isnan(pack["x_test"]).any()


def test_residual_diagnostics():
    """Verify residual metrics computation."""
    actuals = np.array([2.0, 3.0, 4.0, 5.0])
    predictions = np.array([2.1, 2.9, 4.2, 4.8])

    diag = compute_residual_diagnostics(actuals, predictions)
    assert "mean_residual" in diag
    assert "std_residual" in diag
    assert "median_absolute_error" in diag
    assert diag["median_absolute_error"] > 0.0


def test_predictor_input_validation():
    """Verify that predictor clamps out-of-range inputs and provides audit warnings."""
    predictor = ModelPredictor.__new__(ModelPredictor)
    raw_inputs = {
        "T": 85.0,  # exceeds 60C bound
        "RH": -10.0,  # below 0% bound
        "PT08.S1(CO)": "not_a_number",  # invalid type
    }
    cleaned, warnings = predictor.validate_inputs(raw_inputs)
    assert cleaned["T"] == 60.0  # clamped to max bound
    assert cleaned["RH"] == 0.0  # clamped to min bound
    assert "PT08.S1(CO)" not in cleaned
    assert len(warnings) == 3
