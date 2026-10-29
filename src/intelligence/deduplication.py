"""Multi-criteria Event Matching and Deduplication Engine.

Combines semantic embedding similarity, geographic compatibility, publication
temporal proximity, and category alignment to group reports into distinct
real-world pollution events without falsely merging distinct incidents.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import config
from src.intelligence.embeddings import SemanticEmbeddingManager

logger = logging.getLogger(__name__)


def parse_iso_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 string to timezone-aware datetime."""
    if not dt_str:
        return None
    try:
        # Normalize trailing Z to +00:00
        clean = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except Exception as e:
        logger.debug("Failed parsing datetime %s: %s", dt_str, e)
        return None


def are_locations_compatible(loc1: Optional[str], loc2: Optional[str]) -> Tuple[bool, str]:
    """Check geographic compatibility between two extracted location strings.

    Returns (is_compatible, explanation).
    If both specify distinct, non-overlapping cities, returns False (incompatible).
    """
    if not loc1 or not loc2:
        return True, "One or both reports lack location; not geographically conflicting."

    l1 = loc1.strip().lower()
    l2 = loc2.strip().lower()

    if l1 == l2:
        return True, f"Exact location match: '{loc1}'."

    # Substring / subregion check (e.g., 'Noida, Uttar Pradesh, India' and 'Uttar Pradesh, India')
    if l1 in l2 or l2 in l1:
        return True, f"Geographic sub-region overlap between '{loc1}' and '{loc2}'."

    # Specific city conflicts
    cities = [
        "delhi", "mumbai", "kolkata", "chennai", "bengaluru", "hyderabad",
        "ahmedabad", "pune", "coimbatore", "kanpur", "lucknow", "patna",
        "amritsar", "ludhiana", "beijing", "milan"
    ]
    city1 = next((c for c in cities if c in l1), None)
    city2 = next((c for c in cities if c in l2), None)

    if city1 and city2 and city1 != city2:
        return False, f"Geographic conflict: report A refers to '{city1.title()}' while report B refers to '{city2.title()}'."

    return True, "Geographic details do not indicate conflict."


def compute_temporal_delta_hours(time_a: Optional[str], time_b: Optional[str]) -> Optional[float]:
    """Compute absolute difference in hours between two ISO publication timestamps."""
    dt_a = parse_iso_datetime(time_a)
    dt_b = parse_iso_datetime(time_b)

    if not dt_a or not dt_b:
        return None

    delta = abs((dt_a - dt_b).total_seconds()) / 3600.0
    return round(delta, 2)


def match_event_candidates(
    article_a: Dict[str, Any],
    article_b: Dict[str, Any],
    embedding_manager: Optional[SemanticEmbeddingManager] = None,
    similarity_threshold: float = config.SIMILARITY_THRESHOLD,
    max_time_delta_hours: float = config.TEMPORAL_WINDOW_HOURS,
) -> Dict[str, Any]:
    """Evaluate whether two reports refer to the same pollution incident.

    Enforces semantic similarity, location compatibility, and temporal proximity.
    """
    manager = embedding_manager or SemanticEmbeddingManager()

    text_a = f"{article_a.get('title', '')} {article_a.get('content_normalized', '')[:400]}"
    text_b = f"{article_b.get('title', '')} {article_b.get('content_normalized', '')[:400]}"

    semantic_sim = manager.compute_similarity(text_a, text_b)

    # Geographic check
    loc_a = article_a.get("reported_location") or article_a.get("location")
    loc_b = article_b.get("reported_location") or article_b.get("location")
    loc_compatible, loc_reason = are_locations_compatible(loc_a, loc_b)

    # Temporal proximity check
    time_a = article_a.get("published_at") or article_a.get("created_at")
    time_b = article_b.get("published_at") or article_b.get("created_at")
    time_delta = compute_temporal_delta_hours(time_a, time_b)

    time_compatible = True
    time_reason = "Timestamps compatible."
    if time_delta is not None and time_delta > max_time_delta_hours:
        time_compatible = False
        time_reason = f"Temporal distance ({time_delta:.1f} hrs) exceeds {max_time_delta_hours} hr event window."

    # Category compatibility check
    cat_a = article_a.get("category")
    cat_b = article_b.get("category")
    cat_compatible = True
    if cat_a and cat_b and cat_a != cat_b and cat_a != "other_uncertain" and cat_b != "other_uncertain":
        # Disallow merging stubble burning with factory smoke
        if (cat_a == "crop_burning" and cat_b == "industrial_emissions") or (
            cat_a == "industrial_emissions" and cat_b == "crop_burning"
        ):
            cat_compatible = False

    reasons: List[str] = [loc_reason, time_reason]

    # Overall decision
    is_match = False
    combined_score = 0.0

    if not loc_compatible:
        is_match = False
        reasons.append("Rejected: Geographically incompatible reports.")
    elif not time_compatible:
        is_match = False
        reasons.append("Rejected: Outside temporal incident window.")
    elif not cat_compatible:
        is_match = False
        reasons.append(f"Rejected: Conflicting pollution source categories ('{cat_a}' vs '{cat_b}').")
    elif semantic_sim >= similarity_threshold:
        is_match = True
        combined_score = semantic_sim
        reasons.append(f"Matched: Semantic similarity {semantic_sim:.2f} >= threshold {similarity_threshold}.")
    else:
        is_match = False
        reasons.append(f"Rejected: Semantic similarity {semantic_sim:.2f} below threshold {similarity_threshold}.")

    return {
        "is_match": is_match,
        "combined_score": round(combined_score, 4),
        "semantic_similarity": round(semantic_sim, 4),
        "location_compatible": loc_compatible,
        "temporal_delta_hours": time_delta,
        "category_compatible": cat_compatible,
        "reasons": reasons,
    }
