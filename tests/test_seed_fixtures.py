"""Tests for offline fixture database seeder."""

from scripts.seed_offline_fixtures import seed_offline_fixtures, OFFLINE_ARTICLES
from src.storage.repository import PollutionRepository


def test_seed_offline_fixtures_idempotency(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")

    # Initial seed
    count_1 = seed_offline_fixtures(repository=repo)
    assert count_1 == len(OFFLINE_ARTICLES)
    assert repo.count_articles() == len(OFFLINE_ARTICLES)

    # Re-run seeder should be idempotent (no duplicate rows)
    count_2 = seed_offline_fixtures(repository=repo)
    assert count_2 == len(OFFLINE_ARTICLES)
    assert repo.count_articles() == len(OFFLINE_ARTICLES)

    # Verify pollution filter
    pollution_count = repo.count_articles(only_pollution=True)
    assert pollution_count == len(OFFLINE_ARTICLES)
