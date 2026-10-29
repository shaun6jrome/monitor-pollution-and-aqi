"""Unit tests for GDELT API client and timestamp parser."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.gdelt_client import GDELTClient, parse_gdelt_timestamp


def test_parse_gdelt_timestamp_formats():
    """Verify parsing of standard GDELT timestamp formats."""
    # Format: YYYYMMDDTHHMMSSZ
    iso1 = parse_gdelt_timestamp("20241029T143000Z")
    assert iso1 is not None
    assert iso1.startswith("2024-10-29T14:30:00")

    # Format: YYYYMMDDHHMMSS
    iso2 = parse_gdelt_timestamp("20241029143000")
    assert iso2 is not None
    assert iso2.startswith("2024-10-29T14:30:00")

    # Malformed or None
    assert parse_gdelt_timestamp(None) is None
    assert parse_gdelt_timestamp("") is None
    assert parse_gdelt_timestamp("invalid_date") is None


def test_gdelt_fetch_with_mock_response():
    """Verify handling of successful GDELT JSON responses."""
    mock_payload = {
        "articles": [
            {
                "url": "https://example.com/clean-air-study",
                "title": "Study reveals air pollution trends",
                "seendate": "20241028T100000Z",
                "domain": "example.com",
                "language": "en",
            }
        ]
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        client = GDELTClient(cache_dir=Path(tmp_dir))
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = '{"articles": [{"url": "https://example.com/clean-air-study", "title": "Study reveals air pollution trends", "seendate": "20241028T100000Z", "domain": "example.com"}]}'
            mock_resp.json.return_value = mock_payload
            mock_get.return_value = mock_resp

            articles, meta = client.fetch_articles("air pollution", max_records=10, force_refresh=True)
            assert len(articles) == 1
            assert articles[0]["title"] == "Study reveals air pollution trends"
            assert meta["status"] == "LIVE_SUCCESS"


def test_gdelt_fetch_handling_http_error():
    """Verify graceful handling when GDELT returns HTTP errors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        client = GDELTClient(cache_dir=Path(tmp_dir))
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 503
            mock_resp.text = "Service Unavailable"
            mock_get.return_value = mock_resp

            articles, meta = client.fetch_articles("air pollution", force_refresh=True)
            assert len(articles) == 0
            assert meta["status"] == "FAILED"
            assert "503" in meta["error"]
