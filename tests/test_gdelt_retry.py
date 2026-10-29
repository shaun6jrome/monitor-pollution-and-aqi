"""Tests for GDELT API retry behavior, HTTP 429 backoff, and network recovery."""

from unittest.mock import MagicMock, patch
import requests

from src.ingestion.gdelt_client import GDELTClient
from src.storage.repository import PollutionRepository


def test_gdelt_retry_on_http_429(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")
    client = GDELTClient(repository=repo, cache_dir=tmp_path / "cache")

    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.text = "Please limit requests to one every 5 seconds"

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.text = '{"articles": [{"title": "Retry success story", "url": "https://example.com/retry", "seendate": "20250101T000000Z", "domain": "example.com"}]}'
    mock_resp_200.json.return_value = {
        "articles": [
            {
                "title": "Retry success story",
                "url": "https://example.com/retry",
                "seendate": "20250101T000000Z",
                "domain": "example.com",
            }
        ]
    }

    with patch("requests.get", side_effect=[mock_resp_429, mock_resp_200]), \
         patch("time.sleep") as mock_sleep:
        articles, meta = client.fetch_articles("test query", max_records=5, force_refresh=True)

        assert mock_sleep.called
        assert len(articles) == 1
        assert meta["status"] == "LIVE_SUCCESS"
        assert articles[0]["title"] == "Retry success story"


def test_gdelt_retry_on_network_timeout(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")
    client = GDELTClient(repository=repo, cache_dir=tmp_path / "cache")

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.text = '{"articles": [{"title": "Recovered article", "url": "https://example.com/recovered", "seendate": "20250101T000000Z", "domain": "example.com"}]}'
    mock_resp_200.json.return_value = {
        "articles": [
            {
                "title": "Recovered article",
                "url": "https://example.com/recovered",
                "seendate": "20250101T000000Z",
                "domain": "example.com",
            }
        ]
    }

    with patch("requests.get", side_effect=[requests.Timeout("Timeout"), mock_resp_200]), \
         patch("time.sleep") as mock_sleep:
        articles, meta = client.fetch_articles("test query", max_records=5, force_refresh=True)

        assert mock_sleep.called
        assert len(articles) == 1
        assert meta["status"] == "LIVE_SUCCESS"
