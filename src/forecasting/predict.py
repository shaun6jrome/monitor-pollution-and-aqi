"""Inference Pipeline and Input Validation for Air Quality Forecasting.

Validates input feature ranges, handles missing inputs transparently using
training medians, and generates traceable predictions with target units and model metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

import config

logger = logging.getLogger(__name__)

# Valid physical bounds for sensor and meteorological variables
FEATURE_BOUNDS = {
    "T": (-20.0, 60.0),       # Celsius
    "RH": (0.0, 100.0),        # %
    "AH": (0.0, 50.0),         # g/m^3
    "PT08.S1(CO)": (0.0, 3000.0),
    "PT08.S2(NMHC)": (0.0, 3000.0),
    "PT08.S3(NOx)": (0.0, 3000.0),
    "PT08.S4(NO2)": (0.0, 3000.0),
    "PT08.S5(O3)": (0.0, 3000.0),
    "CO(GT)_lag_1": (0.0, 25.0),  # mg/m^3
}


class ModelPredictor:
    """Loads serialized model and serves verified predictions."""

    def __init__(self, model_name: str = "RandomForestRegressor"):
        self.model_name = model_name
        self.artifact_path = config.MODELS_DIR / f"{model_name}.joblib"
        self.pipeline = None
        self.feature_names = []
        self.train_medians = {}
        self.target_col = config.DEFAULT_FORECAST_TARGET
        self.target_unit = "mg/m^3"
        self.metrics = {}
        self.is_loaded = False
        self._load_artifact()

    def _load_artifact(self) -> None:
        """Load joblib artifact from disk."""
        if not self.artifact_path.exists():
            logger.warning("Model artifact not found at %s. Model needs training.", self.artifact_path)
            self.is_loaded = False
            return

        try:
            artifact = joblib.load(self.artifact_path)
            self.pipeline = artifact["pipeline"]
            self.feature_names = artifact["features"]
            self.train_medians = artifact.get("train_medians", {})
            self.target_col = artifact.get("target_col", config.DEFAULT_FORECAST_TARGET)
            self.target_unit = artifact.get("target_unit", "mg/m^3")
            self.metrics = {
                "test_mae": artifact.get("test_mae"),
                "test_rmse": artifact.get("test_rmse"),
                "test_r2": artifact.get("test_r2"),
            }
            self.is_loaded = True
            logger.info("Loaded model %s from %s", self.model_name, self.artifact_path)
        except Exception as e:
            logger.error("Failed to load model artifact %s: %s", self.artifact_path, e)
            self.is_loaded = False

    def validate_inputs(self, raw_inputs: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
        """Validate input values against physical domain bounds.

        Returns (cleaned_inputs, warning_list).
        """
        cleaned = {}
        warnings = []

        for key, val in raw_inputs.items():
            if val is None or val == "":
                continue
            try:
                f_val = float(val)
                if key in FEATURE_BOUNDS:
                    min_b, max_b = FEATURE_BOUNDS[key]
                    if not (min_b <= f_val <= max_b):
                        warnings.append(
                            f"Input '{key}' value {f_val} falls outside expected range [{min_b}, {max_b}]. Clamping to valid range."
                        )
                        f_val = max(min_b, min(max_b, f_val))
                cleaned[key] = f_val
            except (ValueError, TypeError):
                warnings.append(f"Invalid non-numeric input for '{key}': '{val}'. Using training median.")

        return cleaned, warnings

    def predict(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform verified prediction with complete audit trail."""
        if not self.is_loaded or self.pipeline is None:
            return {
                "status": "NOT_LOADED",
                "error": f"Model artifact '{self.model_name}' is not trained or loaded yet.",
                "prediction": None,
            }

        cleaned_inputs, warnings = self.validate_inputs(input_features)

        # Assemble feature vector in exact order
        feature_vector = []
        imputed_features = []

        # Derive time features if hour/month provided
        hour = cleaned_inputs.get("hour", 12)
        month = cleaned_inputs.get("month", 6)
        cleaned_inputs["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
        cleaned_inputs["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
        cleaned_inputs["month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
        cleaned_inputs["month_cos"] = float(np.cos(2 * np.pi * month / 12.0))
        cleaned_inputs["is_weekend"] = float(cleaned_inputs.get("is_weekend", 0))

        for feat in self.feature_names:
            if feat in cleaned_inputs:
                feature_vector.append(cleaned_inputs[feat])
            else:
                # Impute with training median
                median_val = self.train_medians.get(feat, 0.0)
                feature_vector.append(median_val)
                imputed_features.append(feat)

        x_vec = np.array([feature_vector], dtype=np.float64)

        try:
            pred_val = float(self.pipeline.predict(x_vec)[0])
            # Target concentrations cannot be negative physically
            pred_val = max(0.0, pred_val)

            return {
                "status": "SUCCESS",
                "model_name": self.model_name,
                "target_pollutant": self.target_col,
                "predicted_value": round(pred_val, 3),
                "units": self.target_unit,
                "metrics": self.metrics,
                "imputed_features": imputed_features,
                "warnings": warnings,
                "input_summary": cleaned_inputs,
                "explanation": (
                    f"Predicted {self.target_col} concentration is {pred_val:.3f} {self.target_unit} "
                    f"using {self.model_name} (Holdout Test RMSE: {self.metrics.get('test_rmse', 'N/A')})."
                ),
            }
        except Exception as e:
            logger.error("Prediction error: %s", e)
            return {
                "status": "ERROR",
                "error": str(e),
                "prediction": None,
            }
