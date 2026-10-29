"""Tests for official Indian CPCB / data.gov.in air quality connector."""

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.ingestion.cpcb_client import CPCBClient
from src.storage.repository import PollutionRepository


def test_cpcb_unconfigured_without_api_key(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")
    client = CPCBClient(repository=repo, api_key="")

    status = client.check_connection_status()
    assert status["configured"] is False
    assert status["status"] == "NOT_CONFIGURED"
    assert "https://airquality.cpcb.gov.in/AQI_India/" in status["portal_url"]


def test_cpcb_auth_error_on_invalid_key(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")
    client = CPCBClient(repository=repo, api_key="invalid_test_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 403

    with patch("requests.get", return_value=mock_resp):
        status = client.check_connection_status()
        assert status["configured"] is True
        assert status["status"] == "AUTH_ERROR"
        assert "Authentication failed" in status["message"]


def test_cpcb_network_unreachable(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")
    client = CPCBClient(repository=repo, api_key="valid_key")

    with patch("requests.get", side_effect=requests.ConnectionError("Connection failed")):
        status = client.check_connection_status()
        assert status["configured"] is True
        assert status["status"] == "UNREACHABLE"
        assert "Network error" in status["message"]


def test_cpcb_fetch_without_key_returns_safe_notice(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")
    client = CPCBClient(repository=repo, api_key="")

    res = client.fetch_latest_observations()
    assert res["status"] == "KEY_MISSING"
    assert len(res["records"]) == 0
    assert "Official Indian air quality API key not set" in res["message"]


def test_cpcb_successful_fetch_and_storage(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")
    client = CPCBClient(repository=repo, api_key="mock_valid_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "records": [
            {
                "country": "India",
                "state": "Tamil Nadu",
                "city": "Coimbatore",
                "station": "SIDCO Kurichi, Coimbatore - TNPCB",
                "last_update": "29-10-2024 16:00:00",
                "pollutant_id": "PM2.5",
                "pollutant_min": "45",
                "pollutant_max": "112",
                "pollutant_avg": "68",
                "pollutant_unit": "ug/m3",
                "aqi": "145",
            },
            {
                "country": "India",
                "state": "Delhi",
                "city": "Delhi",
                "station": "Anand Vihar, Delhi - DPCC",
                "last_update": "29-10-2024 16:00:00",
                "pollutant_id": "PM10",
                "pollutant_min": "210",
                "pollutant_max": "450",
                "pollutant_avg": "385",
                "pollutant_unit": "ug/m3",
                "aqi": "395",
            },
        ]
    }

    with patch("requests.get", return_value=mock_resp):
        res = client.fetch_latest_observations(limit=5)
        assert res["status"] == "SUCCESS"
        assert len(res["records"]) == 2
        assert res["records"][0]["city"] == "Coimbatore"
        assert res["records"][0]["pollutant"] == "PM2.5"
        assert res["records"][0]["value"] == 68.0

        # Verify stored in repository
        stored = repo.get_latest_cpcb_observations(limit=10)
        assert len(stored) == 2
        assert any(s["station_name"] == "SIDCO Kurichi, Coimbatore - TNPCB" for s in stored)


def test_cpcb_source_metadata():
    client = CPCBClient(api_key="")
    meta = client.get_source_metadata()
    assert meta["source_name"] == "Central Pollution Control Board (CPCB) / data.gov.in"
    assert "PM2.5" in meta["monitored_parameters"]
    assert "data.gov.in" in meta["instructions"]


def test_cpcb_feed_diagnostics():
    client = CPCBClient(api_key="")
    diag = client.diagnose_feed()
    assert diag["configured"] is False
    assert diag["status"] == "NOT_CONFIGURED"
    assert "latency_ms" in diag
    assert diag["latency_ms"] >= 0.0
