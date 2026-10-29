"""Tests for classifier explanation and batch classification utilities."""

import pytest
from src.preprocessing.event_classifier import PollutionClassifier


@pytest.fixture
def classifier():
    return PollutionClassifier()


def test_explain_prediction_positive_pollution(classifier):
    title = "Dense smog blankets Delhi as air quality drops into severe zone"
    content = "High levels of PM2.5 and PM10 recorded across national capital region."
    exp = classifier.explain_prediction(title, content)

    assert exp["label"] == 1
    assert exp["label_name"] == "Pollution-Related"
    assert exp["confidence"] >= 0.55
    assert len(exp["matched_strong_keywords"]) > 0 or len(exp["matched_moderate_keywords"]) > 0
    assert "smog" in exp["matched_strong_keywords"] or "pm2.5" in exp["matched_strong_keywords"]


def test_explain_prediction_negative_politics(classifier):
    title = "Parliament passes banking finance reform bill following floor debate"
    content = "The finance minister outlined monetary policy provisions for commercial lenders."
    exp = classifier.explain_prediction(title, content)

    assert exp["label"] == 0
    assert exp["label_name"] == "Not Pollution"
    assert exp["confidence"] >= 0.55
    assert len(exp["matched_strong_keywords"]) == 0


def test_batch_classify_multiple_articles(classifier):
    articles = [
        {"id": "a1", "title": "Crop burning fires reported in Punjab fields", "content_normalized": "stubble smoke"},
        {"id": "a2", "title": "Stock exchange indexes surge to record highs", "content_normalized": "investor rally"},
    ]
    results = classifier.batch_classify(articles)

    assert len(results) == 2
    assert results[0]["article_id"] == "a1"
    assert results[0]["label"] == 1
    assert results[1]["article_id"] == "a2"
    assert results[1]["label"] == 0
