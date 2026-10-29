"""Data ingestion modules for external sources (GDELT, CPCB, article extraction)."""

from src.ingestion.article_extractor import ArticleExtractor
from src.ingestion.cpcb_client import CPCBClient
from src.ingestion.gdelt_client import GDELTClient

__all__ = ["GDELTClient", "CPCBClient", "ArticleExtractor"]
