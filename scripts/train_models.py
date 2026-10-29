"""CLI Script to Execute End-to-End Model Training and Evaluation.

Trains Linear Regression, Random Forest, and XGBoost on chronological UCI Air Quality
dataset partitions, verifies no leakage, displays MAE and RMSE metrics, and persists artifacts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from scripts.download_dataset import download_uci_dataset
from src.forecasting.evaluate_models import format_model_comparison_table
from src.forecasting.train_models import train_and_evaluate_all_models
from src.storage.repository import PollutionRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate air quality forecasting models")
    parser.add_argument("--target", default=config.DEFAULT_FORECAST_TARGET, help="Target pollutant column")
    parser.add_argument("--download-if-missing", action="store_true", default=True, help="Download UCI dataset if missing")
    args = parser.parse_args()

    # Ensure dataset is available
    if not config.UCI_RAW_CSV_PATH.exists() and args.download_if_missing:
        logger.info("Dataset not detected. Attempting automatic download...")
        success = download_uci_dataset()
        if not success:
            logger.error("Could not obtain dataset. Exiting.")
            sys.exit(1)

    logger.info("Starting model training pipeline for target '%s'...", args.target)
    repo = PollutionRepository()
    results = train_and_evaluate_all_models(target_col=args.target, repository=repo, save_models=True)

    summary_df = format_model_comparison_table(results["models_summary"])

    print("\n" + "=" * 70)
    print("HISTORICAL POLLUTANT CONCENTRATION FORECASTING — EVALUATION RESULTS")
    print(f"Target: {results['target_col']} ({results['target_unit']})")
    print(f"Chronological Train Period: {results['train_period'][0]} to {results['train_period'][1]} ({results['train_samples']} samples)")
    print(f"Chronological Test Period:  {results['test_period'][0]} to {results['test_period'][1]} ({results['test_samples']} samples)")
    print(f"Features: {len(results['features'])} engineered indicators (no temporal lookahead)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print(f"Best Performing Model (Lowest Test RMSE): {results['best_model']}\n")


if __name__ == "__main__":
    main()
