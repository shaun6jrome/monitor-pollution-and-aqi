"""Intelligence modules for semantic embeddings, duplicate detection, provenance, and event discovery."""

from src.intelligence.deduplication import match_event_candidates
from src.intelligence.embeddings import SemanticEmbeddingManager
from src.intelligence.event_aggregator import EventAggregator
from src.intelligence.provenance import analyze_provenance

__all__ = [
    "SemanticEmbeddingManager",
    "match_event_candidates",
    "EventAggregator",
    "analyze_provenance",
]
