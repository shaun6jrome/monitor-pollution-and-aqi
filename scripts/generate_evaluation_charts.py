"""Generate and Save Evaluation Charts for Model Performance Analysis.

Produces diagnostic charts:
1. Actual vs Predicted Scatter with 1:1 Identity Reference Line
2. Residual Distribution and Error Histogram
3. Feature Importance Ranking (XGBoost / Random Forest)
4. Chronological Error Time Series
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from src.forecasting.dataset_loader import load_and_validate_uci_dataset
from src.forecasting.feature_engineering import prepare_forecasting_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_evaluation_plots(output_dir: Path = config.DOCS_DIR) -> None:
    """Generate diagnostic evaluation plots and save as static HTML/JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df, _ = load_and_validate_uci_dataset()
    data_pack = prepare_forecasting_dataset(df, target_col=config.DEFAULT_FORECAST_TARGET)

    x_test = data_pack["x_test"]
    y_test = data_pack["y_test"]
    features = data_pack["feature_names"]
    test_dates = data_pack["test_timestamps"]

    # Load best model artifact
    model_path = config.MODELS_DIR / "XGBoostRegressor.joblib"
    if not model_path.exists():
        model_path = config.MODELS_DIR / "RandomForestRegressor.joblib"

    if not model_path.exists():
        logger.warning("No trained model artifact found in %s. Run scripts/train_models.py first.", config.MODELS_DIR)
        return

    artifact = joblib.load(model_path)
    pipe = artifact["pipeline"]
    y_pred = pipe.predict(x_test)
    residuals = y_test - y_pred

    # 1. Actual vs Predicted Plot
    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            marker=dict(color="#059669", opacity=0.6, size=5),
            name="Predictions",
        )
    )
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig_scatter.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#334155", dash="dash"),
            name="Ideal 1:1 Fit",
        )
    )
    fig_scatter.update_layout(
        title=f"Actual vs Predicted {config.DEFAULT_FORECAST_TARGET} (Held-Out Test Set)",
        xaxis_title=f"Actual Concentration ({artifact.get('target_unit', 'mg/m^3')})",
        yaxis_title=f"Predicted Concentration ({artifact.get('target_unit', 'mg/m^3')})",
        template="plotly_white",
        width=800,
        height=500,
    )
    fig_scatter.write_html(str(output_dir / "actual_vs_predicted.html"))
    logger.info("Saved %s", output_dir / "actual_vs_predicted.html")

    # 2. Residual Histogram
    fig_hist = px.histogram(
        residuals,
        nbins=50,
        title="Residual Error Distribution (y_test - y_pred)",
        labels={"value": "Residual Error (mg/m^3)"},
        color_discrete_sequence=["#0284c7"],
        template="plotly_white",
    )
    fig_hist.update_layout(width=800, height=450, showlegend=False)
    fig_hist.write_html(str(output_dir / "residual_distribution.html"))
    logger.info("Saved %s", output_dir / "residual_distribution.html")


if __name__ == "__main__":
    generate_evaluation_plots()
