"""Model Evaluation Diagnostics and Residual Analysis.

Calculates error distributions, residual diagnostics (bias, variance, skewness),
and structured data frames for Plotly visualization in the ML Laboratory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_residual_diagnostics(actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Calculate statistical properties of prediction residuals (y - y_hat)."""
    residuals = actuals - predictions
    abs_errors = np.abs(residuals)

    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    median_abs_err = float(np.median(abs_errors))
    p95_err = float(np.percentile(abs_errors, 95))
    max_err = float(np.max(abs_errors))

    # Higher-order statistical moments and normality diagnostics
    skewness = 0.0
    kurt = 0.0
    normaltest_stat = None
    normaltest_pvalue = None

    try:
        from scipy import stats
        skewness = float(stats.skew(residuals))
        kurt = float(stats.kurtosis(residuals))
        if len(residuals) >= 20:
            stat_res = stats.normaltest(residuals)
            normaltest_stat = round(float(stat_res.statistic), 4)
            normaltest_pvalue = round(float(stat_res.pvalue), 6)
    except Exception as e:
        logger.warning("Could not compute scipy residual statistics: %s", e)

    return {
        "mean_residual": round(mean_res, 4),
        "std_residual": round(std_res, 4),
        "median_absolute_error": round(median_abs_err, 4),
        "p95_absolute_error": round(p95_err, 4),
        "max_error": round(max_err, 4),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurt, 4),
        "normaltest_statistic": normaltest_stat,
        "normaltest_pvalue": normaltest_pvalue,
    }


def format_model_comparison_table(model_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format comparison table across trained models."""
    rows = []
    for m in model_summaries:
        rows.append(
            {
                "Model": m["model_name"],
                "Test MAE": m["test_mae"],
                "Test RMSE": m["test_rmse"],
                "Test R^2": m.get("test_r2", "N/A"),
                "Train MAE": m["train_mae"],
                "Train RMSE": m["train_rmse"],
                "Status": "Selected Best" if m.get("is_best") else "Alternative",
            }
        )
    return pd.DataFrame(rows).sort_values("Test RMSE").reset_index(drop=True)
