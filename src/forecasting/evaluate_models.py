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

    return {
        "mean_residual": round(mean_res, 4),
        "std_residual": round(std_res, 4),
        "median_absolute_error": round(median_abs_err, 4),
        "p95_absolute_error": round(p95_err, 4),
        "max_error": round(max_err, 4),
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
