"""Evidence Provenance and Syndication Detection.

Evaluates evidence independence across grouped news reports. Distinguishes:
- Independent reporting (novel wording, independent sources)
- Wire syndication / press agency duplication (verbatim or near-verbatim republishing)
- Conflicting report details (conflicting pollutant measurements or event timings)
- Indeterminate independence
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# Known syndication and wire agencies
WIRE_AGENCIES = [
    "press trust of india", "pti", "asian news international", "ani",
    "reuters", "associated press", "ap", "afp", "ians", "uni"
]


def detect_syndication(
    article_a: Dict[str, Any],
    article_b: Dict[str, Any],
    similarity_score: float,
) -> Dict[str, Any]:
    """Assess whether two articles represent syndicated republishing of the same wire story.

    High lexical or semantic similarity (>0.85) across distinct domains strongly suggests
    syndication rather than independent journalistic corroboration.
    """
    domain_a = (article_a.get("source_domain") or "").lower().strip()
    domain_b = (article_b.get("source_domain") or "").lower().strip()
    same_domain = domain_a and domain_b and (domain_a == domain_b)

    text_a = f"{article_a.get('title', '')} {article_a.get('content_normalized', '')}".lower()
    text_b = f"{article_b.get('title', '')} {article_b.get('content_normalized', '')}".lower()

    # Wire agency credits in text
    wires_a = {w for w in WIRE_AGENCIES if f"({w})" in text_a or f"/{w}/" in text_a or f"source: {w}" in text_a}
    wires_b = {w for w in WIRE_AGENCIES if f"({w})" in text_b or f"/{w}/" in text_b or f"source: {w}" in text_b}
    shared_wire = wires_a.intersection(wires_b)

    is_syndicated = False
    relation_type = "independent"
    explanation = ""

    if same_domain:
        relation_type = "same_publisher_update"
        explanation = f"Both articles originate from the same domain ({domain_a}). Likely subsequent update."
    elif shared_wire:
        is_syndicated = True
        relation_type = "wire_syndication"
        wire_name = list(shared_wire)[0].upper()
        explanation = f"Both articles explicitly cite shared wire service '{wire_name}'."
    elif similarity_score >= 0.85:
        is_syndicated = True
        relation_type = "high_similarity_syndication"
        explanation = (
            f"Cross-domain similarity score of {similarity_score:.2f} (>0.85) indicates "
            f"verbatim or near-identical republished wire copy between {domain_a} and {domain_b}."
        )
    elif similarity_score >= 0.65:
        relation_type = "independent_corroboration"
        explanation = (
            f"Moderate semantic similarity ({similarity_score:.2f}) with distinct text phrasing "
            f"indicates independent corroborating reporting by separate outlets."
        )
    else:
        relation_type = "weak_or_divergent"
        explanation = f"Similarity ({similarity_score:.2f}) indicates divergent reporting or distinct incidents."

    return {
        "is_syndicated": is_syndicated,
        "relation_type": relation_type,
        "similarity_score": round(similarity_score, 4),
        "explanation": explanation,
        "domains": [domain_a, domain_b],
    }


def analyze_provenance(articles: List[Dict[str, Any]], sim_matrix: List[List[float]]) -> Dict[str, Any]:
    """Analyze overall evidence independence across a cluster of articles supporting an event."""
    n = len(articles)
    if n <= 1:
        domain = articles[0].get("source_domain", "Unknown") if articles else "None"
        return {
            "distinct_domains": 1 if n == 1 else 0,
            "domain_list": [domain] if n == 1 else [],
            "syndicated_pairs": 0,
            "independent_reports": 1 if n == 1 else 0,
            "evidence_quality_score": 0.50 if n == 1 else 0.0,
            "assessment": "Single source reporting; corroboration not yet established.",
            "notes": "Single report. No independent verification available.",
        }

    domains: Set[str] = set()
    for art in articles:
        dom = (art.get("source_domain") or "unknown").strip().lower()
        if dom:
            domains.add(dom)

    syndicated_count = 0
    independent_count = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i][j] if i < len(sim_matrix) and j < len(sim_matrix[i]) else 0.0
            res = detect_syndication(articles[i], articles[j], score)
            total_pairs += 1
            if res["is_syndicated"]:
                syndicated_count += 1
            else:
                independent_count += 1

    distinct_domain_count = len(domains)
    # Calculate evidence quality: reward distinct domains with independent phrasing
    # Penalize purely syndicated clones masquerading as multiple independent reports
    syndication_ratio = syndicated_count / total_pairs if total_pairs > 0 else 0.0

    if distinct_domain_count >= 3 and syndication_ratio < 0.5:
        quality = 0.90
        assessment = "Strong independent corroboration across multiple distinct news organizations."
    elif distinct_domain_count >= 2 and syndication_ratio < 0.5:
        quality = 0.78
        assessment = "Moderate independent corroboration across separate news organizations."
    elif distinct_domain_count >= 2 and syndication_ratio >= 0.5:
        quality = 0.60
        assessment = "Multiple outlets reported, but evidence indicates widespread wire syndication rather than independent investigations."
    else:
        quality = 0.52
        assessment = "Single domain coverage with multiple related updates."

    return {
        "distinct_domains": distinct_domain_count,
        "domain_list": sorted(list(domains)),
        "syndicated_pairs": syndicated_count,
        "independent_pairs": independent_count,
        "syndication_ratio": round(syndication_ratio, 3),
        "evidence_quality_score": round(quality, 2),
        "assessment": assessment,
        "notes": f"{distinct_domain_count} distinct domain(s) observed across {n} total articles.",
    }
