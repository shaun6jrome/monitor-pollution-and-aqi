"""Unit tests for reporting and export utilities."""

from __future__ import annotations

import json

import pytest

from src.reporting.export import export_articles_to_csv, export_events_to_csv, export_events_to_json, export_models_to_csv


def test_export_events_to_csv_and_json():
    """Verify event export formatting."""
    mock_events = [
        {
            "id": "evt_001",
            "title": "Delhi Smog Spike",
            "category": "general_complaint",
            "reported_location": "Delhi, India",
            "report_count": 3,
            "distinct_domain_count": 2,
            "confidence_score": 0.85,
            "first_published_at": "2024-11-01T10:00:00Z",
            "last_published_at": "2024-11-01T12:00:00Z",
            "status": "active",
        }
    ]

    csv_data = export_events_to_csv(mock_events)
    assert "evt_001" in csv_data
    assert "Delhi Smog Spike" in csv_data
    assert "category" in csv_data.splitlines()[0]

    json_data = export_events_to_json(mock_events)
    parsed = json.loads(json_data)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "evt_001"


def test_export_models_to_csv():
    """Verify model metrics export."""
    mock_models = [
        {
            "id": "run_rf_co",
            "model_type": "RandomForestRegressor",
            "target_pollutant": "CO(GT)",
            "mae": 0.42,
            "rmse": 0.58,
            "r2": 0.84,
            "train_samples": 6000,
            "test_samples": 1500,
            "is_best_model": 1,
            "created_at": "2024-10-29T12:00:00Z",
        }
    ]
    csv_data = export_models_to_csv(mock_models)
    assert "RandomForestRegressor" in csv_data
    assert "0.58" in csv_data
