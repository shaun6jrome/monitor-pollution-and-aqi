"""Model Training, Hyperparameter Configuration, and Evaluation Pipeline.

Trains and rigorously compares three forecasting models:
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. XGBoost Regressor

All models are trained and evaluated on strictly identical chronological test splits.
Metrics (MAE, RMSE, R^2) and artifact paths are persisted to SQLite.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Prevent OpenMP thread collision on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.forecasting.dataset_loader import load_and_validate_uci_dataset
from src.forecasting.feature_engineering import prepare_forecasting_dataset
from src.storage.repository import PollutionRepository

logger = logging.getLogger(__name__)


def get_xgboost_regressor() -> Any:
    """Instantiate XGBoost Regressor or fallback to sklearn GradientBoostingRegressor."""
    try:
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=config.RANDOM_STATE,
            n_jobs=1,
        )
    except Exception as e:
        logger.warning("XGBoost could not be loaded (%s). Using GradientBoostingRegressor fallback.", e)
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=5,
            random_state=config.RANDOM_STATE,
        )


def train_single_model(
    model_name: str,
    estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    use_scaler: bool = False,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train a single model, compute metrics on identical test set, and return evaluation pack."""
    if use_scaler:
        pipe = Pipeline([("scaler", StandardScaler()), ("regressor", estimator)])
    else:
        pipe = estimator

    logger.info("Training %s on %d samples...", model_name, len(x_train))
    pipe.fit(x_train, y_train)

    train_preds = pipe.predict(x_train)
    test_preds = pipe.predict(x_test)

    train_mae = float(mean_absolute_error(y_train, train_preds))
    test_mae = float(mean_absolute_error(y_test, test_preds))

    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))

    test_r2 = float(r2_score(y_test, test_preds))

    # Compute residuals
    residuals = y_test - test_preds

    # Extract feature importance if feature names provided
    feature_importances: Dict[str, float] = {}
    if feature_names:
        reg = pipe.named_steps["regressor"] if use_scaler else pipe
        if hasattr(reg, "feature_importances_"):
            for f_name, imp in zip(feature_names, reg.feature_importances_):
                feature_importances[f_name] = round(float(imp), 4)
        elif hasattr(reg, "coef_"):
            for f_name, c in zip(feature_names, reg.coef_):
                feature_importances[f_name] = round(float(c), 4)

    return {
        "model_name": model_name,
        "pipeline": pipe,
        "train_mae": round(train_mae, 4),
        "test_mae": round(test_mae, 4),
        "train_rmse": round(train_rmse, 4),
        "test_rmse": round(test_rmse, 4),
        "test_r2": round(test_r2, 4),
        "test_predictions": test_preds,
        "test_actuals": y_test,
        "residuals": residuals,
        "feature_importances": feature_importances,
    }


def train_and_evaluate_all_models(
    target_col: str = config.DEFAULT_FORECAST_TARGET,
    repository: Optional[PollutionRepository] = None,
    save_models: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end training and comparative evaluation across all 3 models."""
    repo = repository or PollutionRepository()

    # Load dataset
    df, dataset_meta = load_and_validate_uci_dataset()

    # Feature engineering with chronological train/test split
    data_pack = prepare_forecasting_dataset(df, target_col=target_col)
    x_train = data_pack["x_train"]
    y_train = data_pack["y_train"]
    x_test = data_pack["x_test"]
    y_test = data_pack["y_test"]
    feature_names = data_pack["feature_names"]
    train_period = data_pack["train_period"]
    test_period = data_pack["test_period"]

    models_to_train = [
        ("LinearRegression", LinearRegression(), True),
        (
            "RandomForestRegressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=4,
                random_state=config.RANDOM_STATE,
                n_jobs=1,
            ),
            False,
        ),
        ("XGBoostRegressor", get_xgboost_regressor(), False),
    ]

    results = []
    best_rmse = float("inf")
    best_model_name = ""

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, est, scale in models_to_train:
        res = train_single_model(name, est, x_train, y_train, x_test, y_test, use_scaler=scale, feature_names=feature_names)
        results.append(res)
        if res["test_rmse"] < best_rmse:
            best_rmse = res["test_rmse"]
            best_model_name = name

    # Persist artifacts and log to database
    summary_list = []
    for res in results:
        m_name = res["model_name"]
        is_best = 1 if m_name == best_model_name else 0
        model_path = config.MODELS_DIR / f"{m_name}.joblib"

        if save_models:
            joblib.dump(
                {
                    "pipeline": res["pipeline"],
                    "features": feature_names,
                    "target_col": target_col,
                    "target_unit": data_pack["target_unit"],
                    "train_medians": data_pack["train_medians"],
                    "test_mae": res["test_mae"],
                    "test_rmse": res["test_rmse"],
                    "test_r2": res["test_r2"],
                    "feature_importances": res["feature_importances"],
                },
                model_path,
            )

        run_id = f"run_{m_name}_{target_col}"
        repo.log_model_run(
            run_id=run_id,
            model_type=m_name,
            target_pollutant=target_col,
            train_start=train_period[0],
            train_end=train_period[1],
            test_start=test_period[0],
            test_end=test_period[1],
            train_samples=len(x_train),
            test_samples=len(x_test),
            mae=res["test_mae"],
            rmse=res["test_rmse"],
            r2=res["test_r2"],
            feature_list=feature_names,
            model_path=str(model_path) if save_models else None,
            is_best_model=is_best,
        )

        summary_list.append(
            {
                "model_name": m_name,
                "test_mae": res["test_mae"],
                "test_rmse": res["test_rmse"],
                "test_r2": res["test_r2"],
                "train_mae": res["train_mae"],
                "train_rmse": res["train_rmse"],
                "is_best": bool(is_best),
                "feature_importances": res["feature_importances"],
            }
        )

    return {
        "status": "SUCCESS",
        "target_col": target_col,
        "target_unit": data_pack["target_unit"],
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "train_period": train_period,
        "test_period": test_period,
        "features": feature_names,
        "best_model": best_model_name,
        "models_summary": summary_list,
        "detailed_results": results,
        "dataset_metadata": dataset_meta,
    }
