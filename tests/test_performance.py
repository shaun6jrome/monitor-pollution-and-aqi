"""Performance regression tests ensuring sub-second inference across core modules."""

import time
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.location_extractor import EntityExtractor
from src.preprocessing.event_classifier import PollutionClassifier
from src.storage.repository import PollutionRepository


def test_text_cleaning_sub_millisecond_latency():
    sample = "Delhi recorded severe air quality with PM2.5 levels exceeding 400 ug/m3. " * 5
    t0 = time.perf_counter()
    for _ in range(50):
        _ = clean_text(sample)
    avg_ms = (time.perf_counter() - t0) / 50 * 1000.0
    # Must execute in under 5 milliseconds
    assert avg_ms < 5.0


def test_entity_extraction_latency():
    sample = "Industrial emissions in Ankleshwar and Vapi elevated PM10 and SO2 on Tuesday."
    extractor = EntityExtractor(use_spacy=False)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = extractor.extract_all(sample)
    avg_ms = (time.perf_counter() - t0) / 20 * 1000.0
    # Must execute in under 15 milliseconds
    assert avg_ms < 15.0


def test_classifier_inference_latency():
    clf = PollutionClassifier()
    t0 = time.perf_counter()
    for _ in range(30):
        _ = clf.classify_text("Air pollution smog alert", "PM2.5 hazardous levels reported")
    avg_ms = (time.perf_counter() - t0) / 30 * 1000.0
    # Must execute in under 10 milliseconds
    assert avg_ms < 10.0


def test_sqlite_query_latency(tmp_path):
    repo = PollutionRepository(db_path=tmp_path / "test.db")
    for i in range(10):
        repo.insert_article(url=f"https://example.com/{i}", title=f"Test Article {i}")

    t0 = time.perf_counter()
    for _ in range(20):
        _ = repo.list_articles(limit=10)
    avg_ms = (time.perf_counter() - t0) / 20 * 1000.0
    # SQLite query must return in under 20 milliseconds
    assert avg_ms < 20.0
