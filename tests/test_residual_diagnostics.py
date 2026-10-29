"""Tests for residual statistical diagnostics and normality tests."""

import numpy as np
from src.forecasting.evaluate_models import compute_residual_diagnostics


def test_gaussian_residuals_normality_diagnostics():
    np.random.seed(42)
    # Perfectly normal residuals
    actuals = np.random.normal(loc=10.0, scale=1.0, size=500)
    predictions = actuals + np.random.normal(loc=0.0, scale=0.2, size=500)

    diag = compute_residual_diagnostics(actuals, predictions)
    assert "mean_residual" in diag
    assert "std_residual" in diag
    assert "skewness" in diag
    assert "kurtosis" in diag
    assert "normaltest_pvalue" in diag
    assert abs(diag["skewness"]) < 0.25
    assert diag["normaltest_pvalue"] is not None
    # For Gaussian residuals, p-value should typically not reject normality
    assert diag["normaltest_pvalue"] > 0.01


def test_skewed_residuals_diagnostics():
    np.random.seed(42)
    actuals = np.zeros(200)
    # Exponential right-skewed error
    predictions = -np.random.exponential(scale=2.0, size=200)

    diag = compute_residual_diagnostics(actuals, predictions)
    assert diag["skewness"] > 0.8
    # Normality should be firmly rejected (p < 0.05)
    assert diag["normaltest_pvalue"] < 0.05
