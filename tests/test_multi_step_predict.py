"""Tests for recursive multi-step ahead autoregressive forecasting."""

from unittest.mock import MagicMock
import numpy as np
from src.forecasting.predict import ModelPredictor


def test_predict_multi_step_trajectory():
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.model_name = "MockRegressor"
    predictor.target_col = "CO(GT)"
    predictor.target_unit = "mg/m^3"
    predictor.feature_names = ["hour_sin", "hour_cos", "CO(GT)_lag_1"]
    predictor.train_medians = {"CO(GT)_lag_1": 2.1}
    predictor.metrics = {"test_rmse": 0.42}
    predictor.is_loaded = True

    # Mock pipeline returning value based on input features
    mock_pipe = MagicMock()
    mock_pipe.predict.side_effect = lambda x: np.array([2.5 + 0.1 * x[0, 2]])
    predictor.pipeline = mock_pipe

    initial_inputs = {"hour": 22, "CO(GT)_lag_1": 2.0}
    res = predictor.predict_multi_step(initial_inputs, steps=5)

    assert res["status"] == "SUCCESS"
    assert res["total_steps"] == 5
    assert len(res["forecasts"]) == 5

    # Verify hour progression
    hours = [f["hour"] for f in res["forecasts"]]
    assert hours == [22, 23, 0, 1, 2]

    # Verify all predictions are positive floats
    for f in res["forecasts"]:
        assert isinstance(f["predicted_value"], float)
        assert f["predicted_value"] >= 0.0
        assert f["unit"] == "mg/m^3"


def test_predict_multi_step_unloaded():
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.is_loaded = False
    predictor.pipeline = None
    predictor.model_name = "UnloadedModel"

    res = predictor.predict_multi_step({}, steps=3)
    assert res["status"] == "NOT_LOADED"
    assert len(res["forecasts"]) == 0
