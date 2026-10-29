"""Tests for evidence provenance, wire syndication, and independence auditing."""

from src.intelligence.provenance import detect_syndication, analyze_provenance


def test_detect_explicit_wire_agency_syndication():
    art_a = {
        "source_domain": "thehindu.com",
        "title": "Delhi air quality turns hazardous after Diwali celebrations (PTI)",
        "content_normalized": "Source: PTI. Air quality plunged across monitoring stations.",
    }
    art_b = {
        "source_domain": "deccanherald.com",
        "title": "Delhi air quality turns hazardous after Diwali (PTI)",
        "content_normalized": "Reported by (PTI) as smog blankets city.",
    }
    res = detect_syndication(art_a, art_b, similarity_score=0.72)
    assert res["is_syndicated"] is True
    assert res["relation_type"] == "wire_syndication"
    assert "PTI" in res["explanation"]


def test_detect_high_similarity_wire_copy_without_tag():
    art_a = {
        "source_domain": "outlet1.com",
        "title": "Stubble burning fires cross 3000 in Punjab as smoke drifts south",
        "content_normalized": "Farm fire counts crossed 3000 on Tuesday with plumes drifting.",
    }
    art_b = {
        "source_domain": "outlet2.com",
        "title": "Stubble burning fires cross 3000 in Punjab as smoke drifts south",
        "content_normalized": "Farm fire counts crossed 3000 on Tuesday with plumes drifting.",
    }
    res = detect_syndication(art_a, art_b, similarity_score=0.92)
    assert res["is_syndicated"] is True
    assert res["relation_type"] == "high_similarity_syndication"


def test_detect_independent_corroboration():
    art_a = {
        "source_domain": "tribuneindia.com",
        "title": "Smoke plumes from paddy burning darken skies over Ludhiana",
        "content_normalized": "Agricultural monitoring cells noted hundreds of fire alerts.",
    }
    art_b = {
        "source_domain": "hindustantimes.com",
        "title": "Punjab farm fires spike air pollution across northern plains",
        "content_normalized": "Satellite imagery confirmed dense haze spreading toward NCR.",
    }
    res = detect_syndication(art_a, art_b, similarity_score=0.70)
    assert res["is_syndicated"] is False
    assert res["relation_type"] == "independent_corroboration"


def test_analyze_provenance_cluster():
    articles = [
        {"source_domain": "outlet1.com", "title": "A", "content_normalized": "text"},
        {"source_domain": "outlet2.com", "title": "A (PTI)", "content_normalized": "text (pti)"},
        {"source_domain": "outlet3.com", "title": "B", "content_normalized": "unique text"},
    ]
    sim_matrix = [
        [1.0, 0.90, 0.68],
        [0.90, 1.0, 0.65],
        [0.68, 0.65, 1.0],
    ]
    prov = analyze_provenance(articles, sim_matrix)
    assert prov["distinct_domains"] == 3
    assert prov["syndicated_pairs"] >= 1
    assert "evidence_quality_score" in prov
    assert 0.0 <= prov["evidence_quality_score"] <= 1.0
