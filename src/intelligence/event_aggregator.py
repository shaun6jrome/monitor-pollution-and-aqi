"""Evidence-Aware Event Aggregator and Discovery Pipeline.

Discovers distinct real-world pollution incidents by clustering unstructured
news reports according to multi-dimensional criteria (semantic embeddings,
geographic boundaries, temporal proximity, and source provenance).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import config
from src.intelligence.deduplication import match_event_candidates
from src.intelligence.embeddings import SemanticEmbeddingManager
from src.intelligence.provenance import analyze_provenance
from src.preprocessing.location_extractor import EntityExtractor
from src.storage.repository import PollutionRepository, generate_event_id, get_utc_now_iso

logger = logging.getLogger(__name__)


class EventAggregator:
    """Aggregates unstructured news articles into distinct pollution events."""

    def __init__(
        self,
        repository: Optional[PollutionRepository] = None,
        embedding_manager: Optional[SemanticEmbeddingManager] = None,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
    ):
        self.repository = repository or PollutionRepository()
        self.embedding_manager = embedding_manager or SemanticEmbeddingManager()
        self.entity_extractor = EntityExtractor(use_spacy=False)  # robust regex gazetteer
        self.similarity_threshold = similarity_threshold

    def discover_events_from_articles(
        self,
        articles: Optional[List[Dict[str, Any]]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process unclustered articles and aggregate into distinct events.

        Returns summary of discovered events, total articles processed, and cluster metrics.
        """
        threshold = similarity_threshold or self.similarity_threshold

        if articles is None:
            # Fetch pollution-related articles from database
            articles = self.repository.list_articles(limit=200, only_pollution=True)

        if not articles:
            return {
                "status": "NO_ARTICLES",
                "total_articles": 0,
                "events_created": 0,
                "message": "No pollution-related articles available for clustering.",
            }

        # Sort chronologically by publication time
        sorted_articles = sorted(
            articles,
            key=lambda x: x.get("published_at") or x.get("created_at") or "",
        )

        existing_events = self.repository.list_events(limit=200)
        events_created = 0
        articles_linked = 0

        for art in sorted_articles:
            art_id = art["id"]
            title = art.get("title", "")
            content = art.get("content_normalized") or art.get("content_raw") or ""
            full_text = f"{title} {content}".strip()

            # Extract entities
            entities = self.entity_extractor.extract_all(full_text)
            loc_ent = next((e for e in entities if e["entity_type"] == "LOCATION"), None)
            cat_ent = next((e for e in entities if e["entity_type"] == "SOURCE_CATEGORY"), None)
            pollutant_ents = [e["normalized_value"] for e in entities if e["entity_type"] == "POLLUTANT"]

            loc_name = loc_ent["normalized_value"] if loc_ent else None
            cat_name = cat_ent["normalized_value"] if cat_ent else "other_uncertain"

            # Persist extracted entities to DB if not already present
            existing_entities = self.repository.get_entities_by_article(art_id)
            if not existing_entities:
                for ent in entities:
                    self.repository.insert_entity(
                        article_id=art_id,
                        entity_type=ent["entity_type"],
                        raw_text=ent["raw_text"],
                        normalized_value=ent["normalized_value"],
                        confidence=ent["confidence"],
                        extraction_method=ent["extraction_method"],
                        char_start=ent.get("char_start"),
                        char_end=ent.get("char_end"),
                    )

            # Match against existing events
            best_event_id: Optional[str] = None
            best_score = 0.0

            for ev in existing_events:
                # Prepare synthetic comparison dictionary
                ev_dict = {
                    "title": ev["title"],
                    "content_normalized": ev.get("evidence_summary", ""),
                    "reported_location": ev.get("reported_location"),
                    "published_at": ev.get("last_published_at") or ev.get("first_published_at"),
                    "category": ev.get("category"),
                }
                art_dict = {
                    "title": title,
                    "content_normalized": content[:400],
                    "reported_location": loc_name,
                    "published_at": art.get("published_at") or art.get("created_at"),
                    "category": cat_name,
                }

                match_res = match_event_candidates(
                    art_dict,
                    ev_dict,
                    embedding_manager=self.embedding_manager,
                    similarity_threshold=threshold,
                )

                if match_res["is_match"] and match_res["combined_score"] > best_score:
                    best_score = match_res["combined_score"]
                    best_event_id = ev["id"]

            if best_event_id is not None:
                # Link to existing event
                evidence_snippet = title[:150]
                self.repository.add_article_to_event(
                    event_id=best_event_id,
                    article_id=art_id,
                    similarity_score=best_score,
                    is_primary_source=0,
                    evidence_snippet=evidence_snippet,
                )
                articles_linked += 1

                # Refresh event provenance
                ev_full = self.repository.get_event_with_articles(best_event_id)
                if ev_full and "articles" in ev_full:
                    art_list = ev_full["articles"]
                    sim_mat = [[1.0 if i == j else 0.7 for j in range(len(art_list))] for i in range(len(art_list))]
                    prov = analyze_provenance(art_list, sim_mat)
                    # Update event with provenance and evidence summary
                    # Note: repository handles updating counts automatically
            else:
                # Create brand new distinct event
                pub_time = art.get("published_at") or art.get("created_at") or get_utc_now_iso()
                new_event_id = generate_event_id(cat_name, loc_name, pub_time)

                clean_title = title if len(title) <= 120 else title[:117] + "..."
                evidence = f"Reported: '{title}'. Pollutants noted: {', '.join(pollutant_ents) if pollutant_ents else 'general particulates'}."

                self.repository.create_event(
                    event_id=new_event_id,
                    title=clean_title,
                    category=cat_name,
                    reported_location=loc_name,
                    first_published_at=pub_time,
                    last_published_at=pub_time,
                    report_count=1,
                    distinct_domain_count=1,
                    confidence_score=0.70 if loc_name else 0.50,
                    evidence_summary=evidence,
                    uncertainty_notes="Single initial report. Location unverified by physical ground sensors." if not loc_name else "Location derived from article text; not geocoded.",
                    provenance_summary=f"Initial source: {art.get('source_domain', 'Unknown')}",
                )

                self.repository.add_article_to_event(
                    event_id=new_event_id,
                    article_id=art_id,
                    similarity_score=1.0,
                    is_primary_source=1,
                    evidence_snippet=title[:150],
                )

                events_created += 1
                articles_linked += 1

                # Update in-memory list for subsequent iterations
                existing_events.append(
                    {
                        "id": new_event_id,
                        "title": clean_title,
                        "category": cat_name,
                        "reported_location": loc_name,
                        "first_published_at": pub_time,
                        "last_published_at": pub_time,
                        "evidence_summary": evidence,
                    }
                )

        return {
            "status": "SUCCESS",
            "total_articles": len(sorted_articles),
            "events_created": events_created,
            "articles_linked": articles_linked,
            "active_embedding_method": self.embedding_manager.active_method,
            "distinct_event_count": self.repository.count_events(),
        }
