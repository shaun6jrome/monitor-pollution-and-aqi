"""Data Export and Audit Reporting Module.

Exports discovered pollution events, article provenance logs, and model evaluation
metrics into standardized CSV and JSON formats for external auditing and research reproducibility.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.storage.repository import PollutionRepository

logger = logging.getLogger(__name__)


def export_events_to_csv(events: List[Dict[str, Any]]) -> str:
    """Generate CSV string of discovered events."""
    if not events:
        return "id,title,category,reported_location,report_count,distinct_domain_count,confidence_score,first_published_at,last_published_at\n"

    output = io.StringIO()
    fields = [
        "id",
        "title",
        "category",
        "reported_location",
        "report_count",
        "distinct_domain_count",
        "confidence_score",
        "first_published_at",
        "last_published_at",
        "status",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for ev in events:
        writer.writerow(ev)
    return output.getvalue()


def export_events_to_json(events: List[Dict[str, Any]]) -> str:
    """Generate pretty JSON string of discovered events with evidence."""
    return json.dumps(events, indent=2, default=str)


def export_models_to_csv(model_runs: List[Dict[str, Any]]) -> str:
    """Generate CSV string of recorded model evaluation runs."""
    if not model_runs:
        return "id,model_type,target_pollutant,mae,rmse,r2,train_samples,test_samples,is_best_model\n"

    output = io.StringIO()
    fields = [
        "id",
        "model_type",
        "target_pollutant",
        "mae",
        "rmse",
        "r2",
        "train_samples",
        "test_samples",
        "is_best_model",
        "created_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for m in model_runs:
        writer.writerow(m)
    return output.getvalue()


def export_articles_to_csv(articles: List[Dict[str, Any]]) -> str:
    """Generate CSV string of ingested news articles."""
    if not articles:
        return "id,title,url,source_domain,published_at,is_pollution_related,extraction_status\n"

    output = io.StringIO()
    fields = [
        "id",
        "title",
        "url",
        "source_domain",
        "published_at",
        "search_query",
        "is_pollution_related",
        "classification_confidence",
        "extraction_status",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for art in articles:
        writer.writerow(art)
    return output.getvalue()
