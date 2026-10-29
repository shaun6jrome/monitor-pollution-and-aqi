"""Tests for continuous temporal decay in multi-criteria event matching."""

from src.intelligence.deduplication import match_event_candidates


def test_temporal_decay_penalizes_distant_reports():
    base_art = {
        "title": "Delhi air pollution hits emergency levels after Diwali",
        "content_normalized": "Severe smog enveloped the national capital with PM2.5 over 400.",
        "reported_location": "Delhi, India",
        "published_at": "2024-11-01T12:00:00Z",
        "category": "general_complaint",
    }

    # Near article: 2 hours later
    art_near = {
        "title": "Delhi air pollution hits emergency levels after Diwali",
        "content_normalized": "Severe smog enveloped the national capital with PM2.5 over 400.",
        "reported_location": "Delhi, India",
        "published_at": "2024-11-01T14:00:00Z",
        "category": "general_complaint",
    }

    # Distant article: 60 hours later (within 72h window)
    art_distant = {
        "title": "Delhi air pollution hits emergency levels after Diwali",
        "content_normalized": "Severe smog enveloped the national capital with PM2.5 over 400.",
        "reported_location": "Delhi, India",
        "published_at": "2024-11-04T00:00:00Z",
        "category": "general_complaint",
    }

    res_near = match_event_candidates(base_art, art_near, similarity_threshold=0.60, max_time_delta_hours=72)
    res_distant = match_event_candidates(base_art, art_distant, similarity_threshold=0.60, max_time_delta_hours=72)

    assert res_near["is_match"] is True
    assert res_distant["is_match"] is True
    assert res_near["combined_score"] > res_distant["combined_score"]


def test_missing_timestamp_uses_full_similarity():
    art_a = {
        "title": "Industrial emissions pollute Noyyal river basin",
        "content_normalized": "Chemical effluents discharge noted near Tirupur.",
        "reported_location": "Tirupur, Tamil Nadu, India",
        "category": "industrial_emissions",
    }
    art_b = {
        "title": "Industrial emissions pollute Noyyal river basin",
        "content_normalized": "Chemical effluents discharge noted near Tirupur.",
        "reported_location": "Tirupur, Tamil Nadu, India",
        "category": "industrial_emissions",
    }

    res = match_event_candidates(art_a, art_b, similarity_threshold=0.60)
    assert res["is_match"] is True
    assert res["combined_score"] == res["semantic_similarity"]
