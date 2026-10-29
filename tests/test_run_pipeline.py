"""Tests for end-to-end pipeline runner execution."""

from unittest.mock import MagicMock, patch
import pytest

from scripts.run_pipeline import run_pipeline


def test_run_pipeline_execution_flow(tmp_path):
    mock_gdelt = MagicMock()
    mock_gdelt.ingest_query_results.return_value = {
        "total_retrieved": 10,
        "new_records": 10,
        "status": "LIVE_SUCCESS",
    }

    mock_classifier = MagicMock()
    mock_classifier.classify_text.return_value = {
        "label": 1,
        "confidence": 0.95,
        "method": "tfidf_logistic_regression",
    }

    mock_aggregator = MagicMock()
    mock_aggregator.discover_events_from_articles.return_value = {
        "events_created": 3,
        "total_articles": 10,
    }

    with patch("scripts.run_pipeline.GDELTClient", return_value=mock_gdelt), \
         patch("scripts.run_pipeline.PollutionClassifier", return_value=mock_classifier), \
         patch("scripts.run_pipeline.EventAggregator", return_value=mock_aggregator), \
         patch("scripts.run_pipeline.train_and_evaluate_all_models") as mock_train, \
         patch("scripts.run_pipeline.evaluate_innovation_pipeline") as mock_eval:

        mock_train.return_value = {
            "status": "SUCCESS",
            "models_summary": [
                {
                    "model_name": "XGBoost",
                    "test_mae": 0.25,
                    "test_rmse": 0.40,
                    "test_r2": 0.90,
                    "train_mae": 0.15,
                    "train_rmse": 0.20,
                    "is_best": True,
                }
            ],
            "best_model": "XGBoost",
            "target_col": "CO(GT)",
            "target_unit": "mg/m^3",
        }
        mock_eval.return_value = {
            "comparison_table": [
                {"Method": "Baseline 1", "Discovered Events": 10, "Pairwise F1": 0.50},
                {"Method": "Baseline 2", "Discovered Events": 8, "Pairwise F1": 0.70},
                {"Method": "Proposed", "Discovered Events": 6, "Pairwise F1": 0.95},
            ],
            "metrics_summary": {
                "baseline_keyword": {"event_count": 10, "precision": 0.6, "recall": 0.7, "f1": 0.65},
                "baseline_semantic": {"event_count": 8, "precision": 0.8, "recall": 0.85, "f1": 0.82},
                "proposed_multicriteria": {"event_count": 6, "precision": 0.95, "recall": 0.92, "f1": 0.93},
            },
        }

        # Run pipeline
        run_pipeline(force_refresh_news=False)

        assert mock_gdelt.ingest_query_results.called
        assert mock_aggregator.discover_events_from_articles.called
