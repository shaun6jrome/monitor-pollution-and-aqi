"""Unit tests for SQLite repository and persistence operations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.storage.repository import PollutionRepository, generate_article_id, generate_event_id


@pytest.fixture
def temp_repo():
    """Create a clean isolated SQLite repository in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_pollution.db"
        repo = PollutionRepository(db_path=db_path)
        yield repo


def test_schema_initialization(temp_repo):
    """Verify that all required tables and indexes are initialized."""
    with temp_repo._get_connection() as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "articles" in tables
        assert "entities" in tables
        assert "events" in tables
        assert "event_articles" in tables
        assert "retrieval_logs" in tables
        assert "model_runs" in tables
        assert "cpcb_observations" in tables


def test_article_insertion_and_idempotency(temp_repo):
    """Test idempotent insertion and update of articles."""
    url = "https://example.com/delhi-air-quality-news-2024"
    title = "Delhi Air Quality Worsens"

    art_id1, is_new1 = temp_repo.insert_article(url=url, title=title, search_query="air pollution")
    assert is_new1 is True
    assert art_id1.startswith("art_")

    # Second insertion with identical URL must be flagged as existing
    art_id2, is_new2 = temp_repo.insert_article(url=url, title=title, search_query="air pollution")
    assert is_new2 is False
    assert art_id1 == art_id2

    # Verify retrieval
    article = temp_repo.get_article_by_id(art_id1)
    assert article is not None
    assert article["title"] == title
    assert article["url"] == url


def test_event_creation_and_article_linking(temp_repo):
    """Test creating an event and associating articles with evidence."""
    url = "https://example.com/punjab-stubble-burning"
    art_id, _ = temp_repo.insert_article(url=url, title="Stubble Burning in Punjab")

    event_id = "evt_test_punjab_01"
    temp_repo.create_event(
        event_id=event_id,
        title="Punjab Stubble Burning Incident",
        category="crop_burning",
        reported_location="Punjab, India",
        first_published_at="2024-11-01T10:00:00Z",
        last_published_at="2024-11-01T10:00:00Z",
        evidence_summary="Extensive paddy farm fires recorded across districts.",
    )

    temp_repo.add_article_to_event(
        event_id=event_id,
        article_id=art_id,
        similarity_score=0.92,
        is_primary_source=1,
        evidence_snippet="Fires reported across Sangrur district.",
    )

    event_data = temp_repo.get_event_with_articles(event_id)
    assert event_data is not None
    assert event_data["report_count"] == 1
    assert len(event_data["articles"]) == 1
    assert event_data["articles"][0]["id"] == art_id


def test_retrieval_logging(temp_repo):
    """Verify logging of external ingestion operations."""
    log_id = temp_repo.log_retrieval(
        source_name="GDELT",
        query="industrial emissions",
        status="SUCCESS",
        records_retrieved=25,
        new_records=20,
        duplicate_records=5,
    )
    assert log_id > 0

    latest = temp_repo.get_latest_retrieval_log("GDELT")
    assert latest is not None
    assert latest["records_retrieved"] == 25
    assert latest["new_records"] == 20
    assert latest["status"] == "SUCCESS"
