"""Unit tests for text preprocessing and pollution classification."""

from __future__ import annotations

import pytest

from src.preprocessing.event_classifier import PollutionClassifier
from src.preprocessing.text_cleaner import clean_text, normalize_text, normalize_unicode


def test_unicode_normalization():
    """Verify conversion of curly quotes, non-standard dashes, and HTML entities."""
    raw = "Delhi&apos;s toxic &ldquo;smog&rdquo; &mdash; hazardous levels"
    norm = normalize_unicode(raw)
    assert "'" in norm
    assert '"' in norm
    assert "-" in norm
    assert "&quot;" not in norm
    assert "&mdash;" not in norm


def test_boilerplate_and_url_removal():
    """Verify removal of URLs, email addresses, and boilerplate syndication markers."""
    raw = (
        "Air quality in Delhi reached severe levels.\n"
        "Read more at https://example.com/delhi-air or email editor@news.org\n"
        "Click here to subscribe to our newsletter.\n"
        "Continuous ambient stations recorded PM2.5 above 400 ug/m3."
    )
    cleaned = clean_text(raw)
    assert "https://" not in cleaned
    assert "editor@news.org" not in cleaned
    assert "Click here to subscribe" not in cleaned
    assert "PM2.5 above 400" in cleaned


def test_pollution_classifier_positive_examples():
    """Test classification of unambiguous pollution incident text."""
    classifier = PollutionClassifier()
    text = "Severe toxic smog blankets Delhi as AQI crosses 450 following rampant crop burning in Punjab"
    res = classifier.classify_text(title=text)

    assert res["label"] == 1
    assert res["confidence"] >= 0.60
    assert "pollution" in res["explanation"].lower() or "environmental" in res["explanation"].lower()


def test_pollution_classifier_negative_examples():
    """Test classification of non-pollution news articles."""
    classifier = PollutionClassifier()
    text = "National football league announced schedule for championship finals in sports arena"
    res = classifier.classify_text(title=text)

    assert res["label"] == 0
    assert res["confidence"] >= 0.70


def test_pollution_classifier_empty_input():
    """Test classifier handling of empty or blank inputs."""
    classifier = PollutionClassifier()
    res = classifier.classify_text(title="", content=None)
    assert res["label"] == -1
    assert res["confidence"] == 0.0
