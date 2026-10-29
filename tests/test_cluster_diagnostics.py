"""Tests for cluster quality diagnosis and system clustering statistics."""

from unittest.mock import MagicMock
from src.intelligence.event_aggregator import EventAggregator
from src.storage.repository import PollutionRepository


def test_diagnose_singleton_cluster(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")
    mock_emb = MagicMock()
    agg = EventAggregator(repository=repo, embedding_manager=mock_emb)

    # Create dummy event with 1 article
    ev_id = repo.create_event(event_id="evt_test_1", title="Delhi smog alert", category="smog")
    art_id, _ = repo.insert_article(url="https://example.com/art1", title="Delhi smog alert")
    repo.add_article_to_event(event_id=ev_id, article_id=art_id, similarity_score=1.0)

    diag = agg.diagnose_cluster_quality(ev_id)
    assert diag["status"] == "SINGLETON"
    assert diag["article_count"] == 1
    assert diag["cohesion_score"] == 1.0
    assert diag["is_tight"] is True


def test_diagnose_multi_article_cluster(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")
    mock_emb = MagicMock()
    mock_emb.compute_similarity.return_value = 0.88
    agg = EventAggregator(repository=repo, embedding_manager=mock_emb)

    ev_id = repo.create_event(event_id="evt_test_2", title="Industrial emissions", category="industrial")
    art1, _ = repo.insert_article(url="https://example.com/ind1", title="Plant A emissions")
    art2, _ = repo.insert_article(url="https://example.com/ind2", title="Plant A emissions update")
    repo.add_article_to_event(event_id=ev_id, article_id=art1, similarity_score=1.0)
    repo.add_article_to_event(event_id=ev_id, article_id=art2, similarity_score=0.88)

    diag = agg.diagnose_cluster_quality(ev_id)
    assert diag["status"] == "MULTI_ARTICLE"
    assert diag["article_count"] == 2
    assert diag["average_cohesion"] == 0.88
    assert diag["is_tight"] is True


def test_get_clustering_summary_empty(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")
    mock_emb = MagicMock()
    agg = EventAggregator(repository=repo, embedding_manager=mock_emb)

    summary = agg.get_clustering_summary()
    assert summary["total_events"] == 0
    assert summary["singletons"] == 0
