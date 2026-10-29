"""Tests for model feature importance extraction and validation."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.forecasting.train_models import train_single_model


def test_random_forest_feature_importance_extraction():
    np.random.seed(42)
    x = np.random.randn(100, 4)
    # y strongly depends on feature 0 and 1
    y = 3.0 * x[:, 0] + 1.5 * x[:, 1] + 0.1 * np.random.randn(100)

    feat_names = ["feat_co_lag", "feat_temp", "feat_humidity", "feat_noise"]
    rf = RandomForestRegressor(n_estimators=20, random_state=42, max_depth=4)

    res = train_single_model(
        model_name="TestRF",
        estimator=rf,
        x_train=x[:80],
        y_train=y[:80],
        x_test=x[80:],
        y_test=y[80:],
        feature_names=feat_names,
    )

    imp = res["feature_importances"]
    assert len(imp) == 4
    assert all(k in imp for k in feat_names)
    # Total importance sums to approximately 1.0
    assert abs(sum(imp.values()) - 1.0) < 0.05
    # Feature 0 should have the highest importance
    assert imp["feat_co_lag"] > imp["feat_noise"]


def test_linear_regression_coefficients_extraction():
    np.random.seed(42)
    x = np.random.randn(80, 3)
    y = 2.5 * x[:, 0] - 1.2 * x[:, 1] + 0.1 * np.random.randn(80)

    feat_names = ["var_a", "var_b", "var_c"]
    lr = LinearRegression()

    res = train_single_model(
        model_name="TestLR",
        estimator=lr,
        x_train=x[:60],
        y_train=y[:60],
        x_test=x[60:],
        y_test=y[60:],
        use_scaler=False,
        feature_names=feat_names,
    )

    coefs = res["feature_importances"]
    assert len(coefs) == 3
    assert abs(coefs["var_a"] - 2.5) < 0.2
    assert abs(coefs["var_b"] - (-1.2)) < 0.2
