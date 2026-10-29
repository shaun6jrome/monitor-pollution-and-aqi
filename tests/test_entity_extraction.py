"""Unit tests for entity extraction pipeline."""

from __future__ import annotations

import pytest

from src.preprocessing.location_extractor import EntityExtractor


def test_location_extraction_gazetteer():
    """Verify extraction and normalization of Indian monitoring locations."""
    extractor = EntityExtractor(use_spacy=False)
    text = "Dense smog covered Delhi, Noida, and Gurugram as farm fires flared in Punjab."
    locations = extractor.extract_locations(text)

    extracted_names = [loc["normalized_value"] for loc in locations]
    assert "Delhi, India" in extracted_names
    assert "Noida, Uttar Pradesh, India" in extracted_names
    assert "Punjab, India" in extracted_names


def test_pollutant_extraction():
    """Verify extraction of particulate and chemical pollutants."""
    extractor = EntityExtractor(use_spacy=False)
    text = "Air quality stations recorded alarming spikes in PM2.5, PM10, and high levels of SO2 and NOx."
    pollutants = extractor.extract_pollutants(text)

    pol_names = [p["normalized_value"] for p in pollutants]
    assert "PM2.5" in pol_names
    assert "PM10" in pol_names
    assert "SO2" in pol_names
    assert "NOx" in pol_names


def test_source_category_rules():
    """Verify classification of emission source categories from evidence."""
    extractor = EntityExtractor(use_spacy=False)
    crop_text = "Farmers reported active stubble burning across hundreds of paddy fields."
    categories = extractor.extract_categories(crop_text)

    cat_names = [c["normalized_value"] for c in categories]
    assert "crop_burning" in cat_names


def test_temporal_extraction():
    """Verify extraction of relative and calendar dates."""
    extractor = EntityExtractor(use_spacy=False)
    text = "The toxic emissions were first observed yesterday and worsened on Monday Oct 28."
    temporals = extractor.extract_temporal_expressions(text)

    assert len(temporals) >= 1
    found_raw = [t["raw_text"].lower() for t in temporals]
    assert any("yesterday" in r for r in found_raw) or any("monday" in r for r in found_raw)
