"""Unit tests for multi-criteria deduplication and provenance analysis."""

from __future__ import annotations

import pytest

from src.intelligence.deduplication import are_locations_compatible, compute_temporal_delta_hours, match_event_candidates
from src.intelligence.provenance import detect_syndication


def test_geographic_compatibility():
    """Verify that conflicting cities are marked incompatible while subregions match."""
    compat1, _ = are_locations_compatible("Delhi, India", "Delhi, India")
    assert compat1 is True

    compat2, _ = are_locations_compatible("Noida, Uttar Pradesh, India", "Uttar Pradesh, India")
    assert compat2 is True

    compat3, _ = are_locations_compatible("Delhi, India", "Mumbai, Maharashtra, India")
    assert compat3 is False


def test_temporal_delta_computation():
    """Verify hourly delta calculation between publication dates."""
    t1 = "2024-11-01T12:00:00Z"
    t2 = "2024-11-02T12:00:00Z"
    delta = compute_temporal_delta_hours(t1, t2)
    assert delta == 24.0

    t3 = "2024-11-01T12:00:00Z"
    t4 = "2024-11-20T12:00:00Z"
    delta_long = compute_temporal_delta_hours(t3, t4)
    assert delta_long is not None and delta_long > 72.0


def test_event_matching_rejection_on_conflicting_location():
    """Ensure two reports with identical wording but conflicting cities are NOT merged."""
    art_a = {
        "title": "Severe smog hits city as toxic pollution spikes",
        "content_normalized": "Particulate matter surged to hazardous levels causing breathing difficulties.",
        "reported_location": "Delhi, India",
        "published_at": "2024-11-01T10:00:00Z",
        "category": "general_complaint",
    }
    art_b = {
        "title": "Severe smog hits city as toxic pollution spikes",
        "content_normalized": "Particulate matter surged to hazardous levels causing breathing difficulties.",
        "reported_location": "Mumbai, Maharashtra, India",
        "published_at": "2024-11-01T11:00:00Z",
        "category": "general_complaint",
    }

    match_res = match_event_candidates(art_a, art_b, similarity_threshold=0.60)
    assert match_res["is_match"] is False
    assert any("Geographic conflict" in r or "incompatible" in r for r in match_res["reasons"])


def test_event_matching_rejection_on_temporal_distance():
    """Ensure two reports in the same city separated by weeks are NOT merged into the same incident."""
    art_a = {
        "title": "Severe smog covers Delhi after Diwali",
        "content_normalized": "Thick pollution blanketed the national capital.",
        "reported_location": "Delhi, India",
        "published_at": "2024-11-01T10:00:00Z",
        "category": "general_complaint",
    }
    art_b = {
        "title": "Severe smog covers Delhi after Diwali",
        "content_normalized": "Thick pollution blanketed the national capital.",
        "reported_location": "Delhi, India",
        "published_at": "2024-11-25T10:00:00Z",  # 24 days later
        "category": "general_complaint",
    }

    match_res = match_event_candidates(art_a, art_b, max_time_delta_hours=72)
    assert match_res["is_match"] is False
    assert any("temporal" in r.lower() for r in match_res["reasons"])


def test_syndication_detection():
    """Verify that near-identical copy across distinct domains is flagged as syndication."""
    art_a = {
        "title": "Delhi air quality turns severe (PTI)",
        "content_normalized": "The air quality index in Delhi plunged to severe category reported PTI.",
        "source_domain": "thehindu.com",
    }
    art_b = {
        "title": "Delhi air quality turns severe (PTI)",
        "content_normalized": "The air quality index in Delhi plunged to severe category reported PTI.",
        "source_domain": "ndtv.com",
    }

    res = detect_syndication(art_a, art_b, similarity_score=0.96)
    assert res["is_syndicated"] is True
    assert res["relation_type"] in ("wire_syndication", "high_similarity_syndication")
